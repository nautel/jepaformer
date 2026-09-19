#!/usr/bin/env python3
"""Generate Libri2Mix wav8k/min mix_clean from official LibriMix metadata.

Follows create_librimix_from_metadata.py (gain -> resample -> min length ->
sum) for clean mixtures only, and writes SpeechBrain CSVs.

Usage:
  python tools/make_libri2mix.py --librispeech DIR --metadata DIR \
      --out DIR/Libri2Mix --csv_dir SAVE --sets train-clean-100:train-100 \
      dev-clean:dev test-clean:test --workers 4
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

COLUMNS = ["ID", "duration"] + [
    f"{name}_wav{suffix}"
    for name in ("mix", "s1", "s2", "noise")
    for suffix in ("", "_format", "_opts")
]


def _make_one(job):
    row, librispeech, out_dir = job
    name = row["mixture_ID"] + ".wav"
    paths = {k: os.path.join(out_dir, k, name) for k in ("mix_clean", "s1", "s2")}
    if not all(os.path.exists(p) for p in paths.values()):
        sources = []
        for i in (1, 2):
            audio, rate = sf.read(os.path.join(librispeech, row[f"source_{i}_path"]))
            audio = resample_poly(audio * row[f"source_{i}_gain"], 1, rate // 8000)
            sources.append(audio)
        length = min(len(s) for s in sources)
        sources = [s[:length] for s in sources]
        sf.write(paths["s1"], sources[0], 8000, subtype="FLOAT")
        sf.write(paths["s2"], sources[1], 8000, subtype="FLOAT")
        sf.write(paths["mix_clean"], sources[0] + sources[1], 8000, subtype="FLOAT")
    return paths, sf.info(paths["mix_clean"]).duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv_dir", required=True)
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.csv_dir, exist_ok=True)

    for item in args.sets:
        meta_name, set_name = item.split(":")
        meta = pd.read_csv(os.path.join(args.metadata, f"libri2mix_{meta_name}.csv"))
        out_dir = os.path.join(args.out, "wav8k", "min", set_name)
        for sub in ("mix_clean", "s1", "s2"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        jobs = [(r, args.librispeech, out_dir) for r in meta.to_dict("records")]
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(_make_one, jobs, chunksize=64))

        csv_path = os.path.join(args.csv_dir, f"libri2mix_{set_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            for index, (paths, duration) in enumerate(results):
                row = {"ID": index, "duration": duration}
                for key, path in (("mix", paths["mix_clean"]), ("s1", paths["s1"]),
                                  ("s2", paths["s2"]), ("noise", "")):
                    row.update({f"{key}_wav": path, f"{key}_wav_format": "wav",
                                f"{key}_wav_opts": None})
                writer.writerow(row)
        print(f"{set_name}: {len(results)} mixtures -> {csv_path}")


if __name__ == "__main__":
    main()
