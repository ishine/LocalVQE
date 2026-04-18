/**
 * Benchmark: GGML streaming inference (memory budget, per-frame wall time,
 * op histogram). Consumes 16 kHz WAV pairs through the C API.
 */

#include "localvqe_graph.h"
#include "localvqe_api.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* mic_path   = nullptr;
    const char* ref_path   = nullptr;
    int iters    = 10;
    bool profile = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--in-wav" && i + 2 < argc) {
            mic_path = argv[++i]; ref_path = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--profile") {
            profile = true;
        } else if (!model_path) {
            model_path = argv[i];
        }
    }
    if (!model_path || !mic_path || !ref_path) {
        fprintf(stderr,
            "Usage: bench model.gguf --in-wav mic.wav ref.wav [--iters N] [--profile]\n");
        return 1;
    }

#ifndef LOCALVQE_HAS_SNDFILE
    fprintf(stderr, "Error: bench requires libsndfile\n");
    return 1;
#else
    std::vector<float> mic_pcm = audio_load_mono(mic_path);
    std::vector<float> ref_pcm = audio_load_mono(ref_path);
    if (mic_pcm.empty() || ref_pcm.empty()) return 1;
    int n = (int)std::min(mic_pcm.size(), ref_pcm.size());
    mic_pcm.resize(n);
    ref_pcm.resize(n);

    const int SR = 16000, HOP = 256;
    printf("Input: %d samples (%.2f s)\n", n, n / (float)SR);
    printf("Iterations: %d\n\n", iters);

    if (profile) {
        dvqe_graph_model tmp_m;
        dvqe_stream_graph tmp_sg;
        if (load_graph_model(model_path, tmp_m, false) &&
            build_stream_graph(tmp_m, tmp_sg)) {
            print_memory_budget(tmp_m, tmp_sg);
            putchar('\n');
            print_op_histogram(tmp_sg.graph);
            putchar('\n');
            free_stream_graph(tmp_sg);
            free_graph_model(tmp_m);
        }
    }

    uintptr_t ctx = localvqe_new(model_path);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    std::vector<float> enh(n);

    // Warmup
    localvqe_process_f32(ctx, mic_pcm.data(), ref_pcm.data(), n, enh.data());

    // Timed runs
    std::vector<int64_t> us_total;
    for (int it = 0; it < iters; it++) {
        localvqe_reset(ctx);
        int64_t t0 = ggml_time_us();
        localvqe_process_f32(ctx, mic_pcm.data(), ref_pcm.data(), n, enh.data());
        us_total.push_back(ggml_time_us() - t0);
    }
    std::sort(us_total.begin(), us_total.end());
    double mean = 0;
    for (auto v : us_total) mean += (double)v / us_total.size();
    printf("End-to-end wall time over %d iters: mean=%.1f ms, median=%.1f ms\n",
           iters, mean / 1000.0, (double)us_total[iters/2] / 1000.0);

    int n_frames = n / HOP;
    double ms_per_frame = (mean / 1000.0) / std::max(1, n_frames);
    double frame_budget_ms = 1000.0 * HOP / SR;
    printf("Per-frame: mean=%.3f ms  (%.1f%% of %.3f ms budget,"
           " realtime factor %.2fx)\n",
           ms_per_frame, 100.0 * ms_per_frame / frame_budget_ms,
           frame_budget_ms, frame_budget_ms / std::max(1e-9, ms_per_frame));

    double secs = n / (double)SR;
    printf("Realtime factor (mean): %.2fx on %.2f s of audio\n\n",
           secs / (mean / 1e6), secs);

    localvqe_free(ctx);
    return 0;
#endif
}
