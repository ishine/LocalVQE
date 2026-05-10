#!/usr/bin/env python
"""Regenerate the reference output tensor for one or more checkpoints.

Usage:
    python tests/regenerate_fixtures.py                # all entries
    python tests/regenerate_fixtures.py localvqe-v1.1-1.3M

Run this AFTER you've intentionally changed model output (e.g. fixing a
bug, adding a new published checkpoint). Commit the resulting
fixtures/<name>.out.pt alongside the code change so test_checkpoints.py
keeps passing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
PYTORCH_DIR = HERE.parent
sys.path.insert(0, str(PYTORCH_DIR))

from localvqe.model import LocalVQE  # noqa: E402

# Mirror the constants in test_checkpoints.py — keep these in sync.
_INPUT_SEED = 20260510
_INPUT_LEN = 16000


def _deterministic_input():
    g = torch.Generator().manual_seed(_INPUT_SEED)
    mic = torch.randn(1, _INPUT_LEN, generator=g)
    ref = torch.randn(1, _INPUT_LEN, generator=g)
    return mic, ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", help="checkpoint name (or all if omitted)")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path.cwd() / "checkpoints",
        help="Where checkpoint files live on disk (default: ./checkpoints/)",
    )
    args = parser.parse_args()

    manifest = yaml.safe_load((HERE / "manifest.yaml").read_text())["checkpoints"]
    if args.name:
        manifest = [c for c in manifest if c["name"] == args.name]
        if not manifest:
            sys.exit(f"No manifest entry named {args.name!r}")

    cfg = yaml.safe_load((PYTORCH_DIR / "configs" / "default.yaml").read_text())

    fixtures_dir = HERE / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    for entry in manifest:
        ckpt_path = args.checkpoint_dir / entry["filename"]
        if not ckpt_path.exists():
            print(f"[skip] {entry['name']}: {ckpt_path} not found")
            continue

        model_kwargs = dict(cfg["model"])
        model_kwargs["arch_version"] = entry["arch_version"]
        model = LocalVQE(**model_kwargs, n_freqs=cfg["audio"]["n_freqs"]).eval()

        blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd_key = entry.get("state_dict_key")
        sd = blob[sd_key] if sd_key else blob
        model.load_state_dict(sd, strict=True)
        if entry.get("fold_temperature", False):
            model.align.fold_temperature()

        mic, ref = _deterministic_input()
        with torch.no_grad():
            y = model(mic, ref)

        out_path = fixtures_dir / entry["reference_output"]
        torch.save(y.detach().clone(), out_path)
        print(f"[ok]  {entry['name']}: wrote {out_path} (shape={tuple(y.shape)})")


if __name__ == "__main__":
    main()
