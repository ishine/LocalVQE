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
    int n_frames = n / HOP;

    // Warmup (single clip: exercises first-hop allocs + any lazy init)
    localvqe_process_f32(ctx, mic_pcm.data(), ref_pcm.data(), n, enh.data());

    // Per-hop streaming timings across all iters
    std::vector<int64_t> us_total;
    std::vector<int64_t> us_hop;
    std::vector<std::pair<int,int>> hop_coord;  // (iter, hop-index-within-iter)
    us_hop.reserve((size_t)iters * n_frames);
    hop_coord.reserve((size_t)iters * n_frames);
    for (int it = 0; it < iters; it++) {
        localvqe_reset(ctx);
        int64_t t_clip = ggml_time_us();
        for (int f = 0; f < n_frames; f++) {
            int64_t t0 = ggml_time_us();
            localvqe_process_frame_f32(ctx,
                                       mic_pcm.data() + f * HOP,
                                       ref_pcm.data() + f * HOP,
                                       HOP,
                                       enh.data() + f * HOP);
            us_hop.push_back(ggml_time_us() - t0);
            hop_coord.push_back({it, f});
        }
        us_total.push_back(ggml_time_us() - t_clip);
    }
    std::sort(us_total.begin(), us_total.end());
    // Sort hops by time but keep coordinates alongside by using indices
    std::vector<size_t> hop_idx(us_hop.size());
    for (size_t i = 0; i < hop_idx.size(); i++) hop_idx[i] = i;
    std::sort(hop_idx.begin(), hop_idx.end(),
              [&](size_t a, size_t b) { return us_hop[a] < us_hop[b]; });
    std::vector<int64_t> sorted_us_hop(us_hop.size());
    for (size_t i = 0; i < hop_idx.size(); i++)
        sorted_us_hop[i] = us_hop[hop_idx[i]];
    double mean = 0;
    for (auto v : us_total) mean += (double)v / us_total.size();
    auto pct = [&](double p) {
        size_t i = std::min(sorted_us_hop.size() - 1,
                            (size_t)(p * sorted_us_hop.size() / 100.0));
        return (double)sorted_us_hop[i] / 1000.0;
    };
    double hop_mean = 0;
    for (auto v : us_hop) hop_mean += (double)v / us_hop.size();

    printf("End-to-end wall time over %d iters: mean=%.1f ms, median=%.1f ms\n",
           iters, mean / 1000.0, (double)us_total[iters/2] / 1000.0);

    double frame_budget_ms = 1000.0 * HOP / SR;
    printf("Per-hop (n=%zu): mean=%.3f min=%.3f p50=%.3f p95=%.3f p99=%.3f"
           " max=%.3f ms  (budget %.3f ms)\n",
           us_hop.size(), hop_mean / 1000.0, pct(0), pct(50), pct(95), pct(99),
           pct(100), frame_budget_ms);
    printf("Budget headroom: p50 %.1f%%, p99 %.1f%% of %.3f ms\n",
           100.0 * pct(50) / frame_budget_ms,
           100.0 * pct(99) / frame_budget_ms, frame_budget_ms);

    // Top-10 slowest hops, with (iter, hop-index) coordinates, to see whether
    // outliers cluster at iteration starts (post-reset cold path) or are
    // scattered (suggesting GC / scheduler / shader-cache effects).
    int n_top = std::min((int)us_hop.size(), 10);
    printf("Top %d slowest hops (iter, hop-in-iter): ms\n", n_top);
    for (int i = 0; i < n_top; i++) {
        size_t sorted_idx = us_hop.size() - 1 - i;
        size_t orig = hop_idx[sorted_idx];
        printf("  (%3d, %3d): %.3f ms%s\n",
               hop_coord[orig].first, hop_coord[orig].second,
               (double)us_hop[orig] / 1000.0,
               hop_coord[orig].second == 0 ? "   <-- first hop post-reset" : "");
    }

    double secs = n / (double)SR;
    printf("Realtime factor (mean): %.2fx on %.2f s of audio\n\n",
           secs / (mean / 1e6), secs);

    localvqe_free(ctx);
    return 0;
#endif
}
