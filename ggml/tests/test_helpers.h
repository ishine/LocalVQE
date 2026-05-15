#ifndef LOCALVQE_TEST_HELPERS_H
#define LOCALVQE_TEST_HELPERS_H

// Header-only utilities shared between ggml test binaries. Inline to
// avoid pulling in a separate translation unit just for two helpers.

#include <cstdlib>
#include <string>
#include <sys/stat.h>

namespace localvqe_test {

inline constexpr int kSkipExit = 77;  // CMake SKIP_RETURN_CODE

inline bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// Locate a GGUF by name. Caller may pass an absolute path to bypass the
// search. Otherwise we check $LOCALVQE_GGUF_DIR, then a few build-tree
// relative candidates (bench-assets sits next to the test binary).
inline std::string resolve_gguf(const std::string& explicit_path,
                                const std::string& fname) {
    if (!explicit_path.empty()) return explicit_path;
    if (const char* env = std::getenv("LOCALVQE_GGUF_DIR")) {
        std::string p = std::string(env) + "/" + fname;
        if (file_exists(p)) return p;
    }
    const char* candidates[] = {
        "bench_assets",
        "../bench_assets",
        "../../bench_assets",
    };
    for (const char* c : candidates) {
        std::string p = std::string(c) + "/" + fname;
        if (file_exists(p)) return p;
    }
    return "";
}

}  // namespace localvqe_test

#endif
