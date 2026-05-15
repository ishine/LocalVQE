/**
 * End-to-end regression test for the GGML graph.
 *
 * Loads a published .gguf, runs localvqe_process_f32 on a fixed
 * pre-generated input, and compares against a committed reference
 * output. Intended to mirror the PyTorch tests/test_checkpoints.py
 * test — same role, same SKIP-on-missing semantics.
 *
 * Usage:
 *   test_regression --gguf-name <fname>
 *                   --input <input.f32>
 *                   --expected <expected.f32>
 *                   [--save <output.f32>]
 *                   [--atol 1e-3] [--rtol 1e-2]
 *
 * GGUF resolution order:
 *   1. Absolute path via --gguf <abs_path>
 *   2. $LOCALVQE_GGUF_DIR/<fname>
 *   3. <build>/bench_assets/<fname>  (filled by bench-assets target)
 *
 * Exits 0 on PASS, 1 on FAIL, 77 on SKIP (CMake SKIP_RETURN_CODE).
 *
 * The .f32 fixtures are raw little-endian float32, no header.
 * Layout of <input.f32>: 16000 mic samples followed by 16000 ref
 * samples (32000 floats total). Layout of <expected.f32>: 16000
 * output samples. Regenerate via tests/regenerate_fixtures.py.
 */

#include "localvqe_api.h"
#include "test_helpers.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using localvqe_test::file_exists;
using localvqe_test::resolve_gguf;
static constexpr int SKIP_EXIT = localvqe_test::kSkipExit;
static constexpr int N_SAMPLES = 16000;

static bool read_f32(const std::string& path, std::vector<float>& out, size_t expected_count) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz < 0 || (size_t)sz != expected_count * sizeof(float)) {
        std::fprintf(stderr, "%s: size %ld bytes, expected %zu floats (%zu bytes)\n",
                     path.c_str(), sz, expected_count, expected_count * sizeof(float));
        std::fclose(f);
        return false;
    }
    out.resize(expected_count);
    size_t got = std::fread(out.data(), sizeof(float), expected_count, f);
    std::fclose(f);
    return got == expected_count;
}

static bool write_f32(const std::string& path, const std::vector<float>& data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t wrote = std::fwrite(data.data(), sizeof(float), data.size(), f);
    std::fclose(f);
    return wrote == data.size();
}

int main(int argc, char** argv) {
    std::string gguf_path;
    std::string gguf_name;
    std::string input_path;
    std::string expected_path;
    std::string save_path;
    float atol = 1e-3f;
    float rtol = 1e-2f;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--gguf-name" && i + 1 < argc) gguf_name = argv[++i];
        else if (a == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (a == "--expected" && i + 1 < argc) expected_path = argv[++i];
        else if (a == "--save" && i + 1 < argc) save_path = argv[++i];
        else if (a == "--atol" && i + 1 < argc) atol = std::atof(argv[++i]);
        else if (a == "--rtol" && i + 1 < argc) rtol = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "Unknown arg: %s\n", a.c_str());
            return 1;
        }
    }

    if (input_path.empty()) {
        std::fprintf(stderr, "Missing --input\n");
        return 1;
    }
    if (save_path.empty() && expected_path.empty()) {
        std::fprintf(stderr, "Need either --expected or --save\n");
        return 1;
    }

    std::string resolved = resolve_gguf(gguf_path, gguf_name);
    if (resolved.empty()) {
        std::printf("SKIP: GGUF '%s' not found (set LOCALVQE_GGUF_DIR or build with bench-assets)\n",
                    gguf_name.c_str());
        return SKIP_EXIT;
    }
    if (!file_exists(input_path)) {
        std::fprintf(stderr, "Input fixture missing: %s\n", input_path.c_str());
        return 1;
    }

    // Load input: 16000 mic + 16000 ref.
    std::vector<float> input_buf;
    if (!read_f32(input_path, input_buf, 2 * N_SAMPLES)) return 1;
    const float* mic = input_buf.data();
    const float* ref = input_buf.data() + N_SAMPLES;

    localvqe_ctx_t ctx = localvqe_new(resolved.c_str());
    if (!ctx) {
        std::fprintf(stderr, "localvqe_new failed for %s\n", resolved.c_str());
        return 1;
    }

    std::vector<float> out(N_SAMPLES, 0.0f);
    int ret = localvqe_process_f32(ctx, mic, ref, N_SAMPLES, out.data());
    if (ret != 0) {
        std::fprintf(stderr, "localvqe_process_f32 failed: %s\n", localvqe_last_error(ctx));
        localvqe_free(ctx);
        return 1;
    }
    localvqe_free(ctx);

    if (!save_path.empty()) {
        if (!write_f32(save_path, out)) {
            std::fprintf(stderr, "Failed to write %s\n", save_path.c_str());
            return 1;
        }
        std::printf("Saved %s (%d samples)\n", save_path.c_str(), N_SAMPLES);
        if (expected_path.empty()) return 0;
    }

    // Compare against reference.
    std::vector<float> expected;
    if (!read_f32(expected_path, expected, N_SAMPLES)) return 1;

    float max_abs = 0.0f, sum_abs = 0.0f;
    int n_violations = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        float d = std::fabs(out[i] - expected[i]);
        float tol = atol + rtol * std::fabs(expected[i]);
        if (d > tol) n_violations++;
        if (d > max_abs) max_abs = d;
        sum_abs += d;
    }
    float mean_abs = sum_abs / N_SAMPLES;

    std::printf("max abs diff:  %.3e\n", max_abs);
    std::printf("mean abs diff: %.3e\n", mean_abs);
    std::printf("violations (>atol+rtol*|ref|): %d / %d (atol=%.1e, rtol=%.1e)\n",
                n_violations, N_SAMPLES, atol, rtol);
    if (n_violations == 0) {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("FAIL\n");
    return 1;
}
