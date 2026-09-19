#!/usr/bin/env python3
"""Run a few training steps on random audio to check a configuration.

Example:
  SEQ=mamba MODE=masked L=4 LEN=32000 PREC=fp16 python tools/smoke_test.py
Env: SEQ (transformer|mamba), MODE (none|pointwise|masked), L (sep layers),
LEN (samples), STEPS, PREC (fp32|fp16), DEV, OUT, EXTRA (k=v,...).
"""

import os
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.argv = [sys.argv[0]]

import speechbrain as sb  # noqa: E402
from hyperpyyaml import load_hyperpyyaml  # noqa: E402

import train  # noqa: E402

overrides = {
    "seq_model": os.environ.get("SEQ", "mamba"),
    "jepa_mode": os.environ.get("MODE", "masked"),
    "sep_layers": int(os.environ.get("L", "2")),
    "output_folder": os.environ.get("OUT", "/tmp/jepaformer_smoke"),
    "precision": os.environ.get("PREC", "fp32"),
    "use_speedperturb": False,
    "threshold_byloss": False,
    "health_interval": 1,
}
# EXTRA="n_dp=8,skip_around_intra=False" -> extra YAML overrides
for item in filter(None, os.environ.get("EXTRA", "").split(",")):
    key, value = item.split("=")
    overrides[key] = {"True": True, "False": False}.get(
        value, int(value) if value.isdigit() else value
    )
with open("hparams/jepa2-libri2mix.yaml", encoding="utf-8") as stream:
    hparams = load_hyperpyyaml(stream, overrides)

device = os.environ.get("DEV", "cuda:0" if torch.cuda.is_available() else "cpu")
separator = train.SourcePredictiveSeparation(
    modules=hparams["modules"],
    opt_class=hparams["optimizer"],
    hparams=hparams,
    run_opts={"device": device},
    checkpointer=None,
)
separator.initialize_target_encoder()
separator.on_fit_start()
separator.on_stage_start(sb.Stage.TRAIN, 1)


def millions(module):
    return sum(p.numel() for p in module.parameters()) / 1e6


print(
    "params (M): masknet %.2f, context %.2f, predictor %.2f"
    % (
        millions(separator.modules.masknet),
        millions(separator.modules.context_encoder),
        millions(separator.modules.span_predictor),
    )
)

torch.manual_seed(0)
length = int(os.environ.get("LEN", "8000"))
s1, s2 = torch.randn(2, length) * 0.1, torch.randn(2, length) * 0.1
padded = lambda x: (x, torch.ones(2))  # noqa: E731
batch = types.SimpleNamespace(
    mix_sig=padded(s1 + s2), s1_sig=padded(s1), s2_sig=padded(s2)
)
for step in range(int(os.environ.get("STEPS", "3"))):
    separator.step = step + 1
    loss = separator.fit_batch(batch)
    print(
        f"step {step}: waveform {float(loss):.3f} "
        f"jepa {float(separator.last_source_prediction_loss):.3f}"
    )
print("health:", separator._health)
if torch.cuda.is_available():
    print("peak GPU GB:", torch.cuda.max_memory_allocated() / 1e9)
