# LocalVQE fuzzing

One libFuzzer harness:

- `fuzz_audio` — exercises the audio-processing API
  (`localvqe_process_f32/s16`, the per-frame variants, `localvqe_reset`,
  and the noise-gate setters) with a real GGUF model loaded once and
  reused across iterations.

The GGUF parser is **not** fuzzed. We treat the parsing path as
trusted-input and protect it with a SHA-256 allowlist of released
model files (see `model_hash.cpp`). Any file whose digest isn't on
the allowlist is rejected by `localvqe_new` before reaching
`gguf_init_from_file`, which removes the malicious-file attack surface
that fuzzing would otherwise need to harden.

Built with `-fsanitize=fuzzer,address,undefined` by default.

## Build

```sh
nix develop .#fuzz
cd ggml
cmake -S . -B build-fuzz \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLOCALVQE_FUZZ=ON
cmake --build build-fuzz -j --target fuzz_audio
# Need the model for fuzz_audio; downloads to build-fuzz/bench_assets/.
cmake --build build-fuzz --target regression-assets
```

To drop UBSan (slightly faster), pass
`-DLOCALVQE_FUZZ_SANITIZERS=fuzzer,address`.

## Run

```sh
export LOCALVQE_FUZZ_MODEL=$PWD/build-fuzz/bench_assets/localvqe-v1.2-1.3M-f32.gguf
mkdir -p corpus_audio
./build-fuzz/bin/fuzz_audio -max_len=65536 corpus_audio
```

Any crash repros are written to the current directory as
`crash-<sha>` / `oom-<sha>` / `timeout-<sha>`. Reproduce with:

```sh
./build-fuzz/bin/fuzz_audio crash-<sha>
```

## Notes

- `fuzz_audio` runs at ~10s of execs/sec because each iteration runs
  the full streaming graph; the slow per-input cost is offset by deep
  coverage of the model's forward path. Use `-jobs=N -workers=N` to
  parallelise.
- Sanitizers will flag NaN/Inf propagation as expected behaviour from
  the STFT/IFFT, not bugs. The audio harness squashes those at the
  input boundary.

## Findings + regression tests

| Bug | Status | Test |
|-----|--------|------|
| `memcpy` overlap in `process_frame_graph` (history copies aliased inside gallocr's reuse pool) | **fixed** in `localvqe_graph.cpp` — history copies now route through `hist_scratch` | `fuzz_repro_frame_roundtrip` |
| GGUF parser aborts on malformed metadata (vendored ggml `ggml_abort`) | **mitigated** by the SHA-256 allowlist in `model_hash.cpp`; malformed files never reach the parser | `fuzz_repro_reject_unhashed` |

`fuzz_repro_frame_roundtrip` is meaningful in any build but definitive
under ASan — in plain Release the UB doesn't visibly manifest. Run
the full sanitizer suite via:

```sh
nix develop .#fuzz
cmake -S . -B build-fuzz -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLOCALVQE_FUZZ=ON
cmake --build build-fuzz --target test_fuzz_repros
cd build-fuzz && ctest -R fuzz_repro --output-on-failure
```

## Adding new releases to the allowlist

When releasing a new GGUF, add its SHA-256 to `kAllowed[]` in
`ggml/model_hash.cpp` and mirror the same hash in `CMakeLists.txt`'s
`_dvqe_download` call. The library and the on-disk model are already
coupled by architecture version, so requiring a rebuild for new model
releases is consistent with existing constraints.

For development workflows with locally-built GGUFs that aren't on the
list, set `LOCALVQE_ALLOW_UNHASHED=1` to bypass the check.
