#!/usr/bin/env python
"""Regenerate fixtures for the PyTorch regression tests.

Two fixture types:

  fixtures/<name>.out.pt     — published-checkpoint regression: load the
                               .pt, run forward on the deterministic
                               input, save output. Consumed by
                               test_checkpoints.py.

  fixtures/snapshot_archN.pt — checkpoint-free arch snapshot: build a
                               model with torch.manual_seed(_INIT_SEED),
                               run forward on the same deterministic
                               input, save output. Consumed by
                               test_arch_snapshots.py. The only
                               numerical-drift coverage we have for
                               arch_version=1 (no published v1 .pt).

Usage:
    python tests/regenerate_fixtures.py                  # all entries + snapshots
    python tests/regenerate_fixtures.py localvqe-v1.1-1.3M
    python tests/regenerate_fixtures.py --arch-snapshots-only

Run this AFTER you've intentionally changed model output. Commit the
resulting fixture files alongside the code change.
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

# Mirror the constants in test_checkpoints.py / test_arch_snapshots.py.
_INPUT_SEED = 20260510
_INPUT_LEN = 16000
_INIT_SEED = 1234

ARCH_VERSIONS = (1, 2, 3)


def _deterministic_input():
    g = torch.Generator().manual_seed(_INPUT_SEED)
    mic = torch.randn(1, _INPUT_LEN, generator=g)
    ref = torch.randn(1, _INPUT_LEN, generator=g)
    return mic, ref


def _regen_arch_snapshots(cfg, fixtures_dir):
    for arch_version in ARCH_VERSIONS:
        model_kwargs = dict(cfg["model"])
        model_kwargs["arch_version"] = arch_version
        torch.manual_seed(_INIT_SEED)
        model = LocalVQE(**model_kwargs, n_freqs=cfg["audio"]["n_freqs"]).eval()
        mic, ref = _deterministic_input()
        with torch.no_grad():
            y = model(mic, ref)
        out_path = fixtures_dir / f"snapshot_arch{arch_version}.pt"
        torch.save(y.detach().clone(), out_path)
        print(f"[ok]  snapshot arch={arch_version}: wrote {out_path} (shape={tuple(y.shape)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", help="checkpoint name (or all if omitted)")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path.cwd() / "checkpoints",
        help="Where checkpoint files live on disk (default: ./checkpoints/)",
    )
    parser.add_argument(
        "--arch-snapshots-only",
        action="store_true",
        help="Only regenerate snapshot_archN.pt; skip checkpoint regressions.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load((PYTORCH_DIR / "configs" / "default.yaml").read_text())
    fixtures_dir = HERE / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    if not args.arch_snapshots_only:
        manifest = yaml.safe_load((HERE / "manifest.yaml").read_text())["checkpoints"]
        if args.name:
            manifest = [c for c in manifest if c["name"] == args.name]
            if not manifest:
                sys.exit(f"No manifest entry named {args.name!r}")

        for entry in manifest:
            ckpt_path = args.checkpoint_dir / entry["filename"]
            if not ckpt_path.exists():
                print(f"[skip] {entry['name']}: {ckpt_path} not found")
                continue

            model_kwargs = dict(cfg["model"])
            # Manifest entries pin per-checkpoint geometry (arch_version,
            # dmax, ...). Apply any overlap with default.yaml's model keys.
            for k, v in entry.items():
                if k in model_kwargs:
                    model_kwargs[k] = v
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

    if not args.name:
        _regen_arch_snapshots(cfg, fixtures_dir)


if __name__ == "__main__":
    main()
