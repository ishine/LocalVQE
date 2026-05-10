"""Architecture-level regression tests — no checkpoint required.

These run quickly and verify the *shape* of the model: which keys live
in the state_dict for each `arch_version`, and that random weights
round-trip through `load_state_dict(strict=True)`. Any future change
that drops/renames/adds a parameter will fail one of these tests.
"""
from __future__ import annotations

import io

import pytest
import torch
import yaml

from localvqe.model import LocalVQE


def _load_default_cfg(pytorch_dir):
    return yaml.safe_load((pytorch_dir / "configs" / "default.yaml").read_text())


# Keys the state_dict MUST contain for each arch_version. Picked to cover
# the four mismatch classes from issue #5: encoder norm, residual norm,
# subpixel norm, decoder skip_norm — plus the legacy BN running stats.
EXPECTED_KEYS = {
    1: {
        "mic_enc1.bn.weight",
        "mic_enc1.bn.bias",
        "mic_enc1.bn.running_mean",
        "mic_enc1.bn.running_var",
        "mic_enc1.resblock.bn.weight",
        "mic_enc1.resblock.bn.running_mean",
        "dec2.bn.weight",
        "dec2.bn.running_mean",
    },
    2: {
        "mic_enc1.norm.weight",
        "mic_enc1.norm.bias",
        "mic_enc1.resblock.norm.weight",
        "mic_enc1.resblock.norm.bias",
        "dec1.deconv.norm.weight",
        "dec1.deconv.norm.bias",
        "dec1.skip_norm.weight",
        "dec1.skip_norm.bias",
    },
}

# Keys that MUST NOT exist for a given arch_version (catches accidental
# cross-arch leakage).
FORBIDDEN_KEY_FRAGMENTS = {
    1: (".norm.", ".skip_norm."),
    2: (".bn.", ".running_mean", ".running_var"),
}


@pytest.mark.parametrize("arch_version", [1, 2])
def test_state_dict_keys(arch_version, pytorch_dir):
    cfg = _load_default_cfg(pytorch_dir)
    model_kwargs = dict(cfg["model"])
    model_kwargs["arch_version"] = arch_version
    model = LocalVQE(**model_kwargs, n_freqs=cfg["audio"]["n_freqs"])
    keys = set(model.state_dict().keys())

    missing = EXPECTED_KEYS[arch_version] - keys
    assert not missing, f"arch_version={arch_version} missing keys: {sorted(missing)}"

    forbidden = FORBIDDEN_KEY_FRAGMENTS[arch_version]
    leaked = [k for k in keys if any(frag in k for frag in forbidden)]
    assert not leaked, f"arch_version={arch_version} leaked cross-arch keys: {leaked[:5]}"


@pytest.mark.parametrize("arch_version", [1, 2])
def test_strict_round_trip(arch_version, pytorch_dir):
    """Save random weights, reload with strict=True. Catches any
    parameter that's registered in __init__ but never visited."""
    cfg = _load_default_cfg(pytorch_dir)
    model_kwargs = dict(cfg["model"])
    model_kwargs["arch_version"] = arch_version
    src = LocalVQE(**model_kwargs, n_freqs=cfg["audio"]["n_freqs"])

    buf = io.BytesIO()
    torch.save({"model_state_dict": src.state_dict()}, buf)
    buf.seek(0)
    sd = torch.load(buf, map_location="cpu", weights_only=True)["model_state_dict"]

    dst = LocalVQE(**model_kwargs, n_freqs=cfg["audio"]["n_freqs"])
    dst.load_state_dict(sd, strict=True)


@pytest.mark.parametrize("arch_version", [1, 2])
def test_forward_finite(arch_version, pytorch_dir):
    """Random-weight forward pass produces a finite tensor of the right shape."""
    cfg = _load_default_cfg(pytorch_dir)
    model_kwargs = dict(cfg["model"])
    model_kwargs["arch_version"] = arch_version
    model = LocalVQE(**model_kwargs, n_freqs=cfg["audio"]["n_freqs"]).eval()

    torch.manual_seed(0)
    mic = torch.randn(1, 16000)
    ref = torch.randn(1, 16000)
    with torch.no_grad():
        y = model(mic, ref)

    assert y.shape == (1, cfg["audio"]["n_freqs"], 63, 2)
    assert torch.isfinite(y).all()


def test_arch_version_default_matches_published_checkpoint():
    """If someone bumps the default in LocalVQE.__init__ we want a
    failing test, because the default.yaml and the published .pt
    metadata both target arch_version=2."""
    import inspect
    sig = inspect.signature(LocalVQE.__init__)
    default = sig.parameters["arch_version"].default
    assert default == 2, (
        f"LocalVQE default arch_version is {default}, expected 2 to match "
        "the published localvqe-v1.1-1.3M.pt checkpoint."
    )
