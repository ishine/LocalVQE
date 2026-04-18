# LocalVQE Architecture

LocalVQE is a causal, streaming neural model for joint acoustic echo
cancellation (AEC), noise suppression (NS), and dereverberation of 16 kHz
speech. It follows the overall topology of **DeepVQE** (Indenbom et al.,
Interspeech 2023, [arXiv:2306.03177](https://arxiv.org/abs/2306.03177)) but
redesigns several components for efficient quantized CPU inference.

## Signal Flow

```
 mic PCM ──► DCT-II analysis ──┐
                               ├─► mic encoder (5 blocks) ──┐
 far PCM ──► DCT-II analysis ──┴─► far encoder (2 blocks)   │
                                                            ▼
                                                      AlignBlock
                                                   (soft delay est.)
                                                            │
                                        concat([mic_e2, aligned_far])
                                                            ▼
                                               encoder block 3 (fuse)
                                                            ▼
                                              S4D bottleneck (diagonal
                                                   state-space model)
                                                            ▼
                                                   decoder (5 blocks)
                                                            ▼
                                                    mask head (1×1)
                                                            ▼
                                     CCM (3×3 complex convolving mask)
                                                            ▼
                                               DCT-II synthesis (OLA)
                                                            ▼
                                                       enhanced PCM
```

## Components

| Component | Details |
|-----------|---------|
| Sample rate | 16 kHz |
| Analysis / synthesis | DCT-II, 512-tap Conv1d filterbank, stride 256, frozen orthogonal basis |
| `n_freqs` | 256 |
| Mic encoder | 5 blocks: 2 → 32 → 40 → 40 → 40 → 40 channels |
| Far-end encoder | 2 blocks: 2 → 32 → 40 channels |
| AlignBlock | Cross-attention soft delay, `dmax = 32` (320 ms), `h = 32` similarity channels |
| Encoder block 3 | 80 → 40 (concat mic + aligned far-end) |
| Bottleneck | S4D (diagonal state-space), hidden = 162 |
| Decoder | 5 blocks, sub-pixel upsampling + BatchNorm, mirroring encoder |
| Mask head | 1×1 Conv2d → 27 channels |
| CCM | 27 channels → 3×3 complex convolving mask (real-valued arithmetic) |
| Kernel | (4, 4) time × frequency, causal padding |
| Parameters | ~0.9 M |
| Receptive field | ≈320 ms (AlignBlock) + conv stack |

### DCT-II Analysis/Synthesis

The STFT is replaced with a real-valued DCT-II filterbank expressed as a
frozen `Conv1d(1, 512, kernel_size=512, stride=256)` whose weights are the
orthogonal DCT-II basis. The output is reshaped to `(B, 256, T, 2)` so the
downstream convolutional stack sees the familiar "two channels per bin"
layout of a complex STFT, but every arithmetic operation is real-valued —
this is the main reason LocalVQE maps cleanly onto GGML without needing
complex kernels.

Synthesis uses a learned projection back to 512 samples and a standard
overlap-add reconstruction. Both filterbanks live *inside* the GGML graph;
there is no separate Python / C++ feature-extraction pass.

### AlignBlock

A soft delay-estimation block inspired by the DeepVQE cross-attention
design. A learned similarity head compares mic and reference encoder
activations at `dmax = 32` possible lags (≈320 ms) and produces a
probability distribution over delays; the reference is then shifted by a
weighted combination of these lags before being concatenated into the mic
path. This is differentiable and converges without hard delay labels.

Two bug fixes relative to public DeepVQE re-implementations:

- The original Xiaobin-Rong implementation used `K.shape[1]`
  (hidden = 32) instead of `x_ref.shape[1]` (input channels) for the
  weighted-sum reshape — corrected in `pytorch/localvqe/align.py`.
- The Okrio implementation used `torch.zeros()` without `.to(device)` —
  also fixed here.

### S4D Bottleneck

Replaces DeepVQE's GRU with the diagonal state-space model of
[Gu et al. (S4D, NeurIPS 2022)](https://arxiv.org/abs/2206.11893):

```
h_t = A ⊙ h_{t-1} + B · u_t
y_t = Re(C · h_t) + D · u_t
```

`A` is a diagonal complex matrix parameterized in polar form
(`A = r · exp(j θ)` with `r = exp(-softplus(rate))`) so `|A| < 1` is
guaranteed and training is stable under mixed precision. The recurrence
runs in float32 even inside an autocast region to avoid bfloat16
accumulation drift.

Advantages over GRU for this use case:

- ~50% fewer parameters at the bottleneck (~1.0 M → ~0.5 M scale).
- Channel-wise independent state → amenable to structured pruning.
- All operations reduce to elementwise + dense matmul — no gating
  nonlinearities — which is the cheapest possible fit for GGML.
- Complex diagonal naturally models damped sinusoidal modes relevant to
  speech periodicity.

### CCM (Complex Convolving Mask)

A 3×3 complex convolving mask applied in the DCT-encoded domain, derived
from the NS-only Xiaobin-Rong reference but implemented with real-valued
arithmetic (two real channels per complex number). This matches the GGML
data model exactly and avoids any complex-tensor plumbing.

## Quantization-Friendly Sizing

All tensor dimensions are chosen so that the dominant weight matrices
align with Q4_K (block size 256) and future WHT transforms:

- `n_freqs = 256` (power of 2), giving an encoder spatial halving chain
  256 → 128 → 64 → 32 → 16 → 8.
- Kernel `(4, 4)` → kernel area 16, a power of 2.
- Channel counts and bottleneck width power-of-2 or small multiples.

No quantized weights are published yet — this is runway, not a shipped
feature.

## Performance

Single-stream streaming inference, F32, Zen4 desktop (24 threads):

| Metric | Value |
|---|---|
| Per-frame wall time | ~1.66 ms |
| Frame duration | 16 ms (256 samples @ 16 kHz) |
| Real-time factor | ≈9.6× |
| Weights on disk | ~3.5 MB (F32 GGUF) |

Per-op profiles and ARM measurements are available via
`ggml/build/bin/bench --profile` on your target hardware.

## Training

The published weights were trained with Schedule-Free AdamW
([Defazio et al., NeurIPS 2024](https://arxiv.org/abs/2405.15682)).

## References

- **DeepVQE**: Indenbom, Beltrán, Chernov, Aichner. *DeepVQE: Real Time Deep
  Voice Quality Enhancement for Joint Acoustic Echo Cancellation, Noise
  Suppression and Dereverberation*. Interspeech 2023.
  [arXiv:2306.03177](https://arxiv.org/abs/2306.03177) — upstream
  architecture (mic/far-end encoders, soft-delay cross-attention, decoder,
  complex convolving mask).
- **S4D**: Gu, Gupta, Goel, Ré. *On the Parameterization and Initialization
  of Diagonal State Space Models*. NeurIPS 2022.
  [arXiv:2206.11893](https://arxiv.org/abs/2206.11893) — the diagonal
  state-space formulation used for our bottleneck in place of DeepVQE's
  GRU.
- **Conv-TasNet**: Luo, Mesgarani. *Conv-TasNet: Surpassing Ideal
  Time-Frequency Magnitude Masking for Speech Separation*. IEEE/ACM TASLP
  2019. [arXiv:1809.07454](https://arxiv.org/abs/1809.07454) — lineage of
  replacing STFT with a Conv1d filterbank as the analysis front-end; we
  freeze the basis to the orthogonal DCT-II rather than learning it.
- **Sub-pixel convolution**: Shi, Caballero, Huszár, Totz, Aitken, Bishop,
  Rueckert, Wang. *Real-Time Single Image and Video Super-Resolution Using
  an Efficient Sub-Pixel Convolutional Neural Network*. CVPR 2016.
  [arXiv:1609.05158](https://arxiv.org/abs/1609.05158) — decoder
  upsampling operator (inherited through DeepVQE).
- [Xiaobin-Rong/deepvqe](https://github.com/Xiaobin-Rong/deepvqe) —
  NS-only DeepVQE reference; basis of our real-valued CCM implementation.
- [Okrio/deepvqe](https://github.com/Okrio/deepvqe) — DeepVQE reference
  implementation with the AEC path.
- [GGML](https://github.com/ggml-org/ggml) — tensor library powering the
  C++ inference engine.
