#!/usr/bin/env python
"""Download every checkpoint listed in manifest.yaml from Hugging Face,
SHA256-verified. Mirror of the GGML side's `regression-assets` target.

Usage:
    python tests/fetch_checkpoints.py                    # all entries
    python tests/fetch_checkpoints.py localvqe-v1.1-1.3M # one entry
    python tests/fetch_checkpoints.py --dest <path>      # override dest dir

Default destination is pytorch/checkpoints/. Override with --dest or by
setting LOCALVQE_CKPT_DIR before invoking. Existing files whose SHA256
already matches the manifest are kept as-is (no re-download).

Uses stdlib urllib + hashlib only — no extra deps beyond what
requirements.txt already provides.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PYTORCH_DIR = HERE.parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_sha: str) -> None:
    """Stream URL → dest, hash as we go, verify SHA256, rename atomically."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    print(f"  fetching {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -1
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                if pct != last_pct and pct % 10 == 0:
                    print(f"    {pct}% ({downloaded / 1e6:.1f} MB)")
                    last_pct = pct
    actual = h.hexdigest()
    if actual != expected_sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA256 mismatch for {dest.name}:\n"
            f"  expected {expected_sha}\n  got      {actual}"
        )
    tmp.replace(dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", help="checkpoint name (or all if omitted)")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: $LOCALVQE_CKPT_DIR or pytorch/checkpoints/)",
    )
    args = parser.parse_args()

    dest_dir = args.dest
    if dest_dir is None:
        env = os.environ.get("LOCALVQE_CKPT_DIR")
        dest_dir = Path(env).expanduser() if env else PYTORCH_DIR / "checkpoints"
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest = yaml.safe_load((HERE / "manifest.yaml").read_text())["checkpoints"]
    if args.name:
        manifest = [e for e in manifest if e["name"] == args.name]
        if not manifest:
            sys.exit(f"No manifest entry named {args.name!r}")

    for entry in manifest:
        name = entry["name"]
        url = entry.get("url")
        if not url:
            print(f"[skip] {name}: no url field in manifest")
            continue
        expected = entry["sha256"]
        dest = dest_dir / entry["filename"]
        if dest.exists():
            if _sha256(dest) == expected:
                print(f"[ok]   {name}: already at {dest} (SHA verified)")
                continue
            print(f"[stale] {name}: SHA mismatch, re-downloading")
        else:
            print(f"[get]  {name}: → {dest}")
        _download(url, dest, expected)
        print(f"[ok]   {name}: downloaded and verified")


if __name__ == "__main__":
    main()
