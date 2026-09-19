# JEPAformer

Speech separation with a **training-only, backbone-agnostic JEPA objective**.
A separator (dual-path Transformer = SepFormer, or dual-path Mamba) is trained
with PIT SI-SNR plus a joint-embedding predictive loss; the JEPA head is
dropped at inference, so inference cost equals the plain separator.

Built on SpeechBrain's LibriMix SepFormer recipe and the original
[sepformer-jepa](https://github.com/VietHann/sepformer-jepa) (v1).

## Method

`--jepa_mode`:

| mode | objective |
|---|---|
| `none` | PIT SI-SNR only (baseline) |
| `pointwise` | v1: 1x1 predictor maps PIT-aligned separated latents to EMA(encoder) features of clean sources (cosine) |
| `masked` | v2: span-mask separated latents -> context encoder -> predictor with mask tokens; targets = EMA(encoder + context encoder) on clean sources; L2 to LayerNorm-ed targets on masked frames |

v2 details (`models.py`, `train.py`): cosine EMA momentum schedule
(`target_encoder_momentum` -> `target_encoder_momentum_end`), collapse monitors
(embedding std / effective rank logged in `train_log.txt`), JEPA head
architecture fixed by `jepa_head_model` for a fair backbone ablation.

`--seq_model`: `transformer` (SepFormer blocks) or `mamba` (bidirectional
Mamba blocks, `mamba-ssm`; a slow PyTorch fallback is used if it is missing).

## Setup

```bash
pip install -r requirements.txt   # mamba-ssm needs a matching CUDA build
```

## Data (Libri2Mix, wav8k/min, clean)

Download LibriSpeech `train-clean-100`, `dev-clean`, `test-clean` and the
official LibriMix metadata (`metadata/Libri2Mix/libri2mix_*.csv` from
https://github.com/JorisCos/LibriMix), then:

```bash
python tools/make_libri2mix.py --librispeech /data/LibriSpeech \
  --metadata /data/LibriMix_meta --out /data/Libri2Mix_gen --csv_dir /data/csv \
  --sets test-clean:test dev-clean:dev train-clean-100:train-100 --workers 6
```

LRS2 mixtures: `hparams/sepformer-jepa-lrs2.yaml` + `prepare_data_lrs2.py`.

## Run

```bash
# quick check on random audio
SEQ=mamba MODE=masked L=4 LEN=32000 PREC=fp16 python tools/smoke_test.py

# pilot (1 GPU, nice/ionice)
CSV_DIR=/data/csv tools/run_pilot.sh 0 mamba masked
CSV_DIR=/data/csv tools/run_pilot.sh 1 mamba none

# or directly
python train.py hparams/jepa2-libri2mix.yaml --seq_model=transformer \
  --jepa_mode=masked --train_data=... --valid_data=... --test_data=... --skip_prep=True
```

Original v1 recipe: `hparams/sepformer-jepa-libri2mix.yaml`.
Results: `<output_folder>/train_log.txt` and per-utterance
`test_results.csv` (SDR, SDRi, SI-SNR, SI-SNRi).

## Status

Research code, experiments in progress; no results reported yet.
