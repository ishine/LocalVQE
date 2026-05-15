/**
 * Regression tests for safety / integrity-boundary issues. Each
 * subcommand pins one bug class so it doesn't silently regress.
 *
 * Usage:
 *   test_fuzz_repros frame-roundtrip --gguf-name <name>
 *       Loads a real model and runs one streaming frame. Under ASan,
 *       this surfaced a memcpy-overlap in process_frame_graph (gallocr
 *       placed the in/out history tensors with overlapping byte ranges,
 *       and ggml_backend_tensor_copy reduces to plain memcpy on the CPU
 *       backend). Run as part of an asan build to validate the fix.
 *
 *   test_fuzz_repros reject-unhashed
 *       Writes a small junk buffer to a tmpfile and confirms
 *       localvqe_new rejects it (returns 0) instead of letting the
 *       GGUF parser see attacker-controlled bytes. Verifies the
 *       SHA-256 allowlist boundary in model_hash.cpp.
 *
 * Exits 0 on PASS, 1 on FAIL, 77 on SKIP (CMake SKIP_RETURN_CODE).
 */

#include "localvqe_api.h"
#include "test_helpers.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

using localvqe_test::resolve_gguf;
static constexpr int SKIP_EXIT = localvqe_test::kSkipExit;

// Per-frame history copy must not memcpy aliased ranges — gallocr can
// place persistent in/out tensors at overlapping offsets in its reuse
// pool. ASan catches the UB on the first frame; in plain builds the
// call still exercises the fixed path so the test stays meaningful.
static int cmd_frame_roundtrip(int argc, char** argv) {
    std::string gguf_name;
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gguf-name" && i + 1 < argc) gguf_name = argv[++i];
    }
    if (gguf_name.empty()) {
        std::fprintf(stderr, "frame-roundtrip: missing --gguf-name\n");
        return 1;
    }
    std::string resolved = resolve_gguf("", gguf_name);
    if (resolved.empty()) {
        std::printf("SKIP: GGUF '%s' not found (set LOCALVQE_GGUF_DIR)\n",
                    gguf_name.c_str());
        return SKIP_EXIT;
    }

    localvqe_ctx_t ctx = localvqe_new(resolved.c_str());
    if (!ctx) {
        std::fprintf(stderr, "frame-roundtrip: localvqe_new failed\n");
        return 1;
    }
    int hop = localvqe_hop_length(ctx);
    if (hop <= 0) { localvqe_free(ctx); return 1; }

    std::vector<float> mic(hop, 0.0f), ref(hop, 0.0f), out(hop, 0.0f);
    // Non-zero content so the streaming graph isn't a trivial identity.
    for (int i = 0; i < hop; i++) {
        mic[i] = 0.01f * (i % 17);
        ref[i] = 0.005f * (i % 13);
    }

    int rc = localvqe_process_frame_f32(ctx, mic.data(), ref.data(), hop, out.data());
    localvqe_free(ctx);
    if (rc != 0) {
        std::fprintf(stderr, "frame-roundtrip: process_frame_f32 returned %d\n", rc);
        return 1;
    }
    std::printf("PASS: one streaming frame processed cleanly\n");
    return 0;
}

// localvqe_new MUST reject any file that isn't a released GGUF — the
// hash check is what lets us treat the (unhardened) GGUF parser as
// trusted-input. Writes a tiny junk blob to a tmpfile and confirms
// the loader returns null without reaching gguf_init_from_file.
static int cmd_reject_unhashed(int /*argc*/, char** /*argv*/) {
    char path[] = "/tmp/localvqe_reject_unhashed_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        std::perror("mkstemp");
        return 1;
    }
    const char junk[] = "not a gguf, just some bytes that won't match any "
                        "released-model SHA-256 even by collision miracle.";
    ssize_t w = write(fd, junk, sizeof(junk) - 1);
    close(fd);
    if (w != (ssize_t)(sizeof(junk) - 1)) {
        std::perror("write");
        unlink(path);
        return 1;
    }

    // CTest doesn't scrub env for this entry, so a developer with the
    // bypass exported would silently invalidate the assertion below.
    unsetenv("LOCALVQE_ALLOW_UNHASHED");

    localvqe_ctx_t ctx = localvqe_new(path);
    int rc = 0;
    if (ctx != 0) {
        std::fprintf(stderr,
            "reject-unhashed: loader accepted a non-allowlisted file\n");
        localvqe_free(ctx);
        rc = 1;
    } else {
        std::printf("PASS: non-allowlisted file rejected by hash check\n");
    }
    unlink(path);
    return rc;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <frame-roundtrip|reject-unhashed> [args...]\n", argv[0]);
        return 1;
    }
    std::string sub = argv[1];
    if (sub == "frame-roundtrip") return cmd_frame_roundtrip(argc - 2, argv + 2);
    if (sub == "reject-unhashed") return cmd_reject_unhashed(argc - 2, argv + 2);
    std::fprintf(stderr, "unknown subcommand: %s\n", sub.c_str());
    return 1;
}
