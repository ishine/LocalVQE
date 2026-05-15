// libFuzzer harness for the LocalVQE audio-processing C API.
//
// One real GGUF model is loaded on the first call and reused across
// iterations (loading is far too slow to do per-input). The fuzzer
// drives the per-frame and batch APIs with attacker-controlled audio
// bytes, the s16/f32 conversion path, and the noise-gate setters.
//
// Build via CMake -DLOCALVQE_FUZZ=ON. The model path is taken from the
// LOCALVQE_FUZZ_MODEL env var, with a sane default for in-tree builds.

#include "localvqe_api.h"

#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kMaxFramesPerInput = 16;

localvqe_ctx_t g_ctx = 0;
int g_hop = 0;
int g_fft = 0;

const char* model_path_from_env() {
    if (const char* p = std::getenv("LOCALVQE_FUZZ_MODEL")) return p;
    return "bench_assets/localvqe-v1.2-1.3M-f32.gguf";
}

void init_once() {
    if (g_ctx) return;
    const char* path = model_path_from_env();
    g_ctx = localvqe_new(path);
    if (!g_ctx) {
        std::fprintf(stderr,
            "fuzz_audio: failed to load model '%s' (set LOCALVQE_FUZZ_MODEL)\n",
            path);
        std::abort();
    }
    g_hop = localvqe_hop_length(g_ctx);
    g_fft = localvqe_fft_size(g_ctx);
    if (g_hop <= 0 || g_fft <= 0) std::abort();
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int* /*argc*/, char*** /*argv*/) {
    init_once();
    return 0;
}

// Input layout (consumed greedily; short inputs are padded with zeros):
//   byte 0     : op selector (low 3 bits)
//   byte 1     : n_samples scaler (in units of hop) — clamped to [0, 16]
//   byte 2     : noise-gate enabled flag
//   bytes 3..6 : noise-gate threshold (float32, little-endian)
//   bytes 7..  : audio payload (mic interleaved with ref)
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    init_once();

    uint8_t op           = size > 0 ? data[0] : 0;
    uint8_t hops_byte    = size > 1 ? data[1] : 0;
    uint8_t gate_enabled = size > 2 ? data[2] : 0;
    float   gate_thresh  = -45.0f;
    if (size >= 7) std::memcpy(&gate_thresh, data + 3, sizeof(float));

    // Reset between iterations so OLA / history state doesn't leak.
    localvqe_reset(g_ctx);
    localvqe_set_noise_gate(g_ctx, gate_enabled & 1, gate_thresh);
    int gate_out_enabled = 0;
    float gate_out_thresh = 0.0f;
    localvqe_get_noise_gate(g_ctx, &gate_out_enabled, &gate_out_thresh);

    const uint8_t* payload = size > 7 ? data + 7 : nullptr;
    size_t payload_size    = size > 7 ? size - 7 : 0;

    auto fill_f32 = [&](std::vector<float>& v) {
        for (size_t i = 0; i < v.size(); i++) {
            size_t off = (i * 4) % (payload_size ? payload_size : 1);
            uint32_t bits = 0;
            if (payload && payload_size >= 4) {
                size_t a = off;
                size_t b = (off + 1) % payload_size;
                size_t c = (off + 2) % payload_size;
                size_t d = (off + 3) % payload_size;
                bits = (uint32_t)payload[a]
                     | ((uint32_t)payload[b] << 8)
                     | ((uint32_t)payload[c] << 16)
                     | ((uint32_t)payload[d] << 24);
            }
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            // Squash NaN/Inf — the API documents [-1, 1] f32 input. NaN
            // propagation across STFT/IFFT isn't an interesting bug class
            // for us (it's well-defined; the model output is just NaN).
            if (!(f == f) || f >  1e30f) f =  1.0f;
            if (              f < -1e30f) f = -1.0f;
            v[i] = f;
        }
    };
    auto fill_s16 = [&](std::vector<int16_t>& v) {
        for (size_t i = 0; i < v.size(); i++) {
            size_t off = (i * 2) % (payload_size ? payload_size : 1);
            uint16_t bits = 0;
            if (payload && payload_size >= 2) {
                size_t a = off;
                size_t b = (off + 1) % payload_size;
                bits = (uint16_t)payload[a] | ((uint16_t)payload[b] << 8);
            }
            v[i] = (int16_t)bits;
        }
    };

    int n_batch = std::min((int)hops_byte, kMaxFramesPerInput) * g_hop;
    switch (op & 0x7) {
        case 0: {
            std::vector<float> mic(g_hop), ref(g_hop), out(g_hop);
            fill_f32(mic); fill_f32(ref);
            localvqe_process_frame_f32(g_ctx, mic.data(), ref.data(),
                                       g_hop, out.data());
            break;
        }
        case 1: {
            std::vector<int16_t> mic(g_hop), ref(g_hop), out(g_hop);
            fill_s16(mic); fill_s16(ref);
            localvqe_process_frame_s16(g_ctx, mic.data(), ref.data(),
                                       g_hop, out.data());
            break;
        }
        case 2: {
            // Drive the batch API with sub-n_fft sizes too, to exercise
            // the short-input early return.
            std::vector<float> mic(n_batch), ref(n_batch), out(n_batch);
            fill_f32(mic); fill_f32(ref);
            localvqe_process_f32(g_ctx, mic.data(), ref.data(), n_batch, out.data());
            break;
        }
        case 3: {
            std::vector<int16_t> mic(n_batch), ref(n_batch), out(n_batch);
            fill_s16(mic); fill_s16(ref);
            localvqe_process_s16(g_ctx, mic.data(), ref.data(), n_batch, out.data());
            break;
        }
        case 4: {
            // Wrong hop size — must be rejected, not OOB-read.
            int bogus = (int)hops_byte;
            if (bogus == g_hop) bogus++;
            std::vector<float> mic(g_hop), ref(g_hop), out(g_hop);
            fill_f32(mic); fill_f32(ref);
            localvqe_process_frame_f32(g_ctx, mic.data(), ref.data(),
                                       bogus, out.data());
            break;
        }
        case 5: {
            std::vector<float> mic(g_hop), ref(g_hop), out(g_hop);
            fill_f32(mic); fill_f32(ref);
            localvqe_process_frame_f32(g_ctx, mic.data(), ref.data(),
                                       g_hop, out.data());
            localvqe_reset(g_ctx);
            localvqe_process_frame_f32(g_ctx, mic.data(), ref.data(),
                                       g_hop, out.data());
            break;
        }
        case 6: {
            std::vector<float>   micf(g_hop), reff(g_hop), outf(g_hop);
            std::vector<int16_t> mics(g_hop), refs(g_hop), outs(g_hop);
            fill_f32(micf); fill_f32(reff);
            fill_s16(mics); fill_s16(refs);
            int frames = (hops_byte & (kMaxFramesPerInput - 1)) + 1;
            for (int i = 0; i < frames; i++) {
                if (i & 1) {
                    localvqe_process_frame_s16(g_ctx, mics.data(), refs.data(),
                                               g_hop, outs.data());
                } else {
                    localvqe_process_frame_f32(g_ctx, micf.data(), reff.data(),
                                               g_hop, outf.data());
                }
            }
            break;
        }
        default: {
            (void)localvqe_sample_rate(g_ctx);
            (void)localvqe_hop_length(g_ctx);
            (void)localvqe_fft_size(g_ctx);
            (void)localvqe_last_error(g_ctx);
            break;
        }
    }
    return 0;
}
