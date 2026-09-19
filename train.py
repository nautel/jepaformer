#!/usr/bin/env python3
"""Train SepFormer with source-predictive latent supervision on Libri2Mix."""

import importlib.util
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEECHBRAIN_ROOT = Path(
    os.environ.get("SPEECHBRAIN_ROOT", PROJECT_ROOT / "speechbrain")
)
BASE_RECIPE_DIR = SPEECHBRAIN_ROOT / "recipes" / "LibriMix" / "separation"

sys.path.insert(0, str(SPEECHBRAIN_ROOT))
sys.path.insert(0, str(BASE_RECIPE_DIR))

import speechbrain as sb  # noqa: E402
from speechbrain.nnet.losses import PitWrapper, cal_si_snr  # noqa: E402
from speechbrain.utils.distributed import run_on_main  # noqa: E402
from speechbrain.utils.logger import get_logger  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import embedding_health, ema_momentum, sample_span_mask  # noqa: E402


def _load_baseline_recipe():
    recipe_path = BASE_RECIPE_DIR / "train.py"
    spec = importlib.util.spec_from_file_location(
        "speechbrain_librimix_sepformer_recipe", recipe_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SpeechBrain recipe from {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline_recipe = _load_baseline_recipe()
logger = get_logger(__name__)


def negative_si_snr(predictions, targets):
    """Return negative SI-SNR for prediction-target pairs."""
    return cal_si_snr(targets, predictions)


class SourcePredictiveSeparation(baseline_recipe.Separation):
    """SepFormer training with PIT-aligned clean-source feature prediction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pit_si_snr = PitWrapper(negative_si_snr)
        momentum = self.hparams.target_encoder_momentum
        if not 0.0 <= momentum < 1.0:
            raise ValueError("target_encoder_momentum must be in [0, 1)")
        self.jepa_mode = self.hparams.jepa_mode
        if self.jepa_mode not in ("none", "pointwise", "masked"):
            raise ValueError("jepa_mode must be none, pointwise or masked")
        self._freeze_target_encoder()
        self._reset_training_metrics()

    def _ema_pairs(self):
        """(online, target) module pairs updated by EMA."""
        pairs = [("encoder", "target_encoder")]
        if self.jepa_mode == "masked":
            pairs.append(("context_encoder", "target_context_encoder"))
        return pairs

    def _reset_training_metrics(self):
        self._source_loss_sum = 0.0
        self._total_loss_sum = 0.0
        self._metric_item_count = 0
        self._optimizer_steps = 0
        self._skipped_optimizer_steps = 0
        self._health = None

    def _update_training_metrics(self, total_loss, source_loss):
        item_count = total_loss.numel()
        self._total_loss_sum += total_loss.detach().float().sum().item()
        self._source_loss_sum += source_loss.detach().float().sum().item()
        self._metric_item_count += item_count

    def _training_metric_stats(self):
        count = max(1, self._metric_item_count)
        return {
            "source-prediction-loss": self._source_loss_sum / count,
            "total-loss": self._total_loss_sum / count,
            "optimizer-steps": self._optimizer_steps,
            "skipped-optimizer-steps": self._skipped_optimizer_steps,
            **(self._health or {}),
        }

    def on_stage_start(self, stage, epoch=None):
        super().on_stage_start(stage, epoch)
        if stage == sb.Stage.TRAIN:
            self._reset_training_metrics()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        super().on_stage_end(stage, stage_loss, epoch)
        if stage == sb.Stage.TRAIN:
            self.train_stats.update(self._training_metric_stats())

    def _freeze_target_encoder(self):
        for _, target_name in self._ema_pairs():
            self.modules[target_name].requires_grad_(False)
            self.modules[target_name].eval()

    def initialize_target_encoder(self):
        """Initialize the frozen target networks from the online networks."""
        for online_name, target_name in self._ema_pairs():
            self.modules[target_name].load_state_dict(
                self.modules[online_name].state_dict()
            )
        self._freeze_target_encoder()

    def current_momentum(self):
        return ema_momentum(
            self.optimizer_step,
            self.hparams.ema_anneal_steps,
            self.hparams.target_encoder_momentum,
            self.hparams.target_encoder_momentum_end,
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """Update the JEPA target networks from the online networks by EMA."""
        momentum = self.current_momentum()
        for online_name, target_name in self._ema_pairs():
            online = self.modules[online_name]
            target = self.modules[target_name]
            online = getattr(online, "module", online)
            target = getattr(target, "module", target)
            for target_parameter, online_parameter in zip(
                target.parameters(), online.parameters()
            ):
                target_parameter.mul_(momentum).add_(
                    online_parameter, alpha=1.0 - momentum
                )
            for target_buffer, online_buffer in zip(
                target.buffers(), online.buffers()
            ):
                target_buffer.copy_(online_buffer)
            target.eval()

    def _prepare_training_inputs(self, mix, targets, stage, noise=None):
        mix, mix_lens = mix
        mix = mix.to(self.device)
        mix_lens = mix_lens.to(self.device)
        valid_samples = torch.ceil(mix_lens * mix.shape[1]).long()
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                # torchaudio's CPU resampler is numerically unsafe under
                # bfloat16 autocast for long waveforms.
                with torch.autocast(
                    device_type=mix.device.type, enabled=False
                ):
                    if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                        previous_length = mix.shape[1]
                        mix, targets = self.add_speed_perturb(
                            targets, mix_lens
                        )
                        mix = targets.sum(-1)
                        length_scale = mix.shape[1] / previous_length
                        valid_samples = torch.ceil(
                            valid_samples.float() * length_scale
                        ).long()

                        if self.hparams.use_wham_noise:
                            noise = noise.to(self.device)
                            min_len = min(noise.shape[1], mix.shape[1])
                            mix = mix[:, :min_len] + noise[:, :min_len]
                            targets = targets[:, :min_len, :]
                            valid_samples = valid_samples.clamp_max(min_len)

                    if self.hparams.use_wavedrop:
                        relative_lengths = valid_samples.float() / mix.shape[1]
                        mix = self.hparams.drop_chunk(mix, relative_lengths)
                        mix = self.hparams.drop_freq(mix)

                    if self.hparams.limit_training_signal_len:
                        crop_length = self.hparams.training_signal_len
                        crop_start = torch.randint(
                            0,
                            1 + max(0, mix.shape[1] - crop_length),
                            (1,),
                        ).item()
                        crop_end = crop_start + crop_length
                        mix = mix[:, crop_start:crop_end]
                        targets = targets[:, crop_start:crop_end, :]
                        valid_samples = (valid_samples - crop_start).clamp(
                            min=0, max=mix.shape[1]
                        )

        valid_samples = valid_samples.clamp(min=0, max=mix.shape[1])
        return mix, valid_samples, targets

    def _valid_latent_frames(self, valid_samples, total_frames):
        encoder = self.modules.target_encoder.conv1d
        kernel = encoder.kernel_size[0]
        stride = encoder.stride[0]
        dilation = encoder.dilation[0]
        padding = encoder.padding[0]
        effective_kernel = dilation * (kernel - 1) + 1
        valid_frames = (
            torch.div(
                valid_samples + (2 * padding) - effective_kernel,
                stride,
                rounding_mode="floor",
            )
            + 1
        )
        return valid_frames.clamp(min=0, max=total_frames)

    def compute_forward_with_latents(self, mix, targets, stage, noise=None):
        """Separate a mixture and expose source and clean-target latents."""
        mix, valid_samples, targets = self._prepare_training_inputs(
            mix, targets, stage, noise
        )

        mixture_latent = self.modules.encoder(mix)
        estimated_masks = self.modules.masknet(mixture_latent)
        repeated_mixture = torch.stack([mixture_latent] * self.hparams.num_spks)
        separated_latents = repeated_mixture * estimated_masks

        estimated_sources = torch.cat(
            [
                self.modules.decoder(separated_latents[index]).unsqueeze(-1)
                for index in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        original_length = mix.size(1)
        estimated_length = estimated_sources.size(1)
        if original_length > estimated_length:
            estimated_sources = F.pad(
                estimated_sources,
                (0, 0, 0, original_length - estimated_length),
            )
        else:
            estimated_sources = estimated_sources[:, :original_length, :]

        batch, samples, sources = targets.shape
        clean_waveforms = targets.permute(0, 2, 1).reshape(
            batch * sources, samples
        )
        with torch.no_grad():
            self.modules.target_encoder.eval()
            clean_latents = self.modules.target_encoder(clean_waveforms)
        clean_latents = clean_latents.reshape(
            batch,
            sources,
            clean_latents.shape[1],
            clean_latents.shape[2],
        )

        valid_frames = self._valid_latent_frames(
            valid_samples, clean_latents.shape[-1]
        )
        separated_latents = separated_latents.permute(1, 0, 2, 3)
        return (
            estimated_sources,
            targets,
            separated_latents,
            clean_latents,
            valid_frames,
        )

    def compute_training_objectives(
        self,
        estimated_sources,
        targets,
        separated_latents,
        clean_latents,
        valid_frames,
    ):
        """Compute waveform PIT and source prediction with one assignment."""
        waveform_loss, permutations = self.pit_si_snr(
            estimated_sources, targets
        )
        if self.jepa_mode == "none":
            return waveform_loss, waveform_loss, torch.zeros_like(waveform_loss)

        latents_for_reordering = separated_latents.permute(0, 2, 3, 1)
        aligned_latents = self.pit_si_snr.reorder_tensor(
            latents_for_reordering, permutations
        ).permute(0, 3, 1, 2)
        if self.jepa_mode == "pointwise":
            predicted_clean_latents = self.modules.source_predictor(
                aligned_latents
            )
            source_prediction_loss = self.hparams.source_prediction_loss(
                predicted_clean_latents,
                clean_latents,
                valid_frames,
            )
        else:
            source_prediction_loss = self._masked_prediction_loss(
                aligned_latents, clean_latents, valid_frames
            )

        total_loss = waveform_loss + (
            self.hparams.source_prediction_weight * source_prediction_loss
        )
        return total_loss, waveform_loss, source_prediction_loss

    def _masked_prediction_loss(self, aligned_latents, clean_latents,
                                valid_frames):
        """JEPA v2: predict contextual clean-source embeddings of masked spans.

        Online: PIT-aligned separated latents -> span masking (frames zeroed)
        -> context encoder -> predictor with mask tokens.
        Target: EMA(encoder) on clean sources -> EMA(context encoder).
        """
        batch, sources, channels, frames = aligned_latents.shape
        flat = aligned_latents.reshape(batch * sources, channels, frames)
        frame_mask = sample_span_mask(
            batch * sources,
            frames,
            self.hparams.mask_ratio,
            self.hparams.mask_span,
            flat.device,
        )
        visible = flat * (~frame_mask).unsqueeze(1).to(flat.dtype)
        context = self.modules.context_encoder(visible)
        predictions = self.modules.span_predictor(context, frame_mask)
        predictions = predictions.reshape(batch, sources, channels, frames)

        with torch.no_grad():
            self.modules.target_context_encoder.eval()
            targets = self.modules.target_context_encoder(
                clean_latents.reshape(batch * sources, channels, frames)
            ).reshape(batch, sources, channels, frames)

        frame_index = torch.arange(frames, device=flat.device)
        valid = frame_index.unsqueeze(0) < valid_frames.to(flat.device).unsqueeze(1)
        active = clean_latents.detach().float().norm(dim=2) > 1e-4
        weight = frame_mask.reshape(batch, sources, frames).float()
        weight = weight + self.hparams.visible_loss_weight * (1 - weight)
        weight = weight * (valid.unsqueeze(1) & active).float()

        if self.step % self.hparams.health_interval == 0:
            pred_std, pred_rank = embedding_health(predictions)
            tgt_std, tgt_rank = embedding_health(targets)
            self._health = {
                "pred-std": pred_std,
                "pred-rank": pred_rank,
                "target-std": tgt_std,
                "target-rank": tgt_rank,
                "ema-momentum": self.current_momentum(),
            }
        return self.hparams.masked_prediction_loss(predictions, targets, weight)

    def fit_batch(self, batch):
        """Train one batch and exclude already-solved waveform examples."""
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        noise = batch.noise_sig[0] if self.hparams.use_wham_noise else None

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with self.training_ctx:
            forward_outputs = self.compute_forward_with_latents(
                mixture, targets, sb.Stage.TRAIN, noise
            )
            total_loss, waveform_loss, source_loss = (
                self.compute_training_objectives(*forward_outputs)
            )

            losses_are_finite = (
                torch.isfinite(waveform_loss).all()
                and torch.isfinite(source_loss).all()
                and torch.isfinite(total_loss).all()
            )
            if not losses_are_finite:
                self.nonfinite_count += 1
                self._skipped_optimizer_steps += 1
                logger.warning(
                    "Non-finite loss: waveform=%s source=%s total=%s "
                    "(%s batches this epoch)",
                    waveform_loss.detach().cpu().tolist(),
                    source_loss.detach().cpu().tolist(),
                    total_loss.detach().cpu().tolist(),
                    self.nonfinite_count,
                )
                self.optimizer.zero_grad()
                return torch.zeros(())

            if self.hparams.threshold_byloss:
                selected = waveform_loss > self.hparams.threshold
                total_loss = total_loss[selected]
                waveform_loss = waveform_loss[selected]
                source_loss = source_loss[selected]

            if total_loss.nelement() == 0:
                self._skipped_optimizer_steps += 1
                self.optimizer.zero_grad()
                return torch.zeros(())

            self._update_training_metrics(total_loss, source_loss)
            optimization_loss = total_loss.mean()
            reported_loss = waveform_loss.mean()

        if optimization_loss < self.hparams.loss_upper_lim:
            scale_before_step = self.scaler.get_scale()
            self.scaler.scale(optimization_loss).backward()
            if self.hparams.clip_grad_norm >= 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(), self.hparams.clip_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < scale_before_step:
                self._skipped_optimizer_steps += 1
            else:
                self._optimizer_steps += 1
                self.optimizer_step += 1
                self.update_target_encoder()
        else:
            self._skipped_optimizer_steps += 1
            logger.warning(
                "Skipped excessive training loss: %s",
                optimization_loss.detach().item(),
            )

        self.optimizer.zero_grad()
        self.last_source_prediction_loss = source_loss.mean().detach()
        return reported_loss.detach().cpu()


def _load_libri_preparation():
    prep_path = Path(
        "/project/anhlt/lab-ss/LongHorn-TasNet-Libri/utils/prepare_data_libri.py"
    )
    spec = importlib.util.spec_from_file_location(
        "longhorn_librimix_preparation", prep_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LibriMix preparation from {prep_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_datasets(hparams):
    # With skip_prep the CSVs already exist (e.g. from tools/make_libri2mix.py).
    if hparams["skip_prep"] and not hparams["dynamic_mixing"]:
        return baseline_recipe.dataio_prep(hparams)
    libri_preparation = _load_libri_preparation()

    run_on_main(
        libri_preparation.prepare_wsjmix,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
            "librimix_addnoise": hparams["use_wham_noise"],
            "fs": hparams["sample_rate"],
        },
    )

    if not hparams["dynamic_mixing"]:
        return baseline_recipe.dataio_prep(hparams)

    from dynamic_mixing import dynamic_mix_data_prep_librimix

    if "processed" not in hparams["base_folder_dm"]:
        processed_folder = (
            os.path.normpath(hparams["base_folder_dm"]) + "_processed"
        )
        if not os.path.exists(processed_folder):
            from recipes.LibriMix.meta.preprocess_dynamic_mixing import (
                resample_folder,
            )

            run_on_main(
                resample_folder,
                kwargs={
                    "input_folder": hparams["base_folder_dm"],
                    "output_folder": processed_folder,
                    "fs": hparams["sample_rate"],
                    "regex": "**/*.flac",
                },
            )
        hparams["base_folder_dm"] = processed_folder

    dynamic_hparams = {
        "train_data": hparams["train_data"],
        "data_folder": hparams["data_folder"],
        "base_folder_dm": hparams["base_folder_dm"],
        "sample_rate": hparams["sample_rate"],
        "num_spks": hparams["num_spks"],
        "training_signal_len": hparams["training_signal_len"],
        "dataloader_opts": hparams["dataloader_opts"],
    }
    train_data = dynamic_mix_data_prep_librimix(dynamic_hparams)
    _, valid_data, test_data = baseline_recipe.dataio_prep(hparams)
    return train_data, valid_data, test_data


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as stream:
        hparams = load_hyperpyyaml(stream, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        raise ValueError(
            "base_folder_dm must exist when dynamic_mixing is enabled"
        )

    if run_opts.get("device") == "cpu" and hparams.get("precision") == "fp16":
        hparams["precision"] = "bf16"

    train_data, valid_data, test_data = _prepare_datasets(hparams)

    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    separator = SourcePredictiveSeparation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Mamba layers rely on their own dt/A initialization, so they are not reset.
    if "pretrained_separator" not in hparams and hparams["seq_model"] != "mamba":
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    separator.initialize_target_encoder()
    separator.fit(
        separator.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )
    separator.evaluate(test_data, min_key="si-snr")
    if not separator.debug:
        separator.save_results(test_data)


if __name__ == "__main__":
    main()
