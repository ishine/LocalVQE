#ifndef LOCALVQE_API_H
#define LOCALVQE_API_H

/**
 * LocalVQE C API — purego-compatible shared library interface.
 *
 * All functions use simple C types (no structs, no variadic args)
 * for compatibility with Go's purego FFI.
 *
 * Typical usage:
 *   uintptr_t ctx = localvqe_new("model.gguf");
 *   if (!ctx) { handle error }
 *   int ret = localvqe_process_f32(ctx, mic, ref, n_samples, out);
 *   localvqe_free(ctx);
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #ifdef LOCALVQE_BUILD
    #define LOCALVQE_API __declspec(dllexport)
  #else
    #define LOCALVQE_API __declspec(dllimport)
  #endif
#else
  #define LOCALVQE_API __attribute__((visibility("default")))
#endif

/**
 * Create a new LocalVQE context by loading a GGUF model file.
 * Returns an opaque handle, or 0 on failure.
 */
LOCALVQE_API uintptr_t localvqe_new(const char* model_path);

/**
 * Free a LocalVQE context and all associated resources.
 */
LOCALVQE_API void localvqe_free(uintptr_t ctx);

/**
 * Process audio through the AEC model (float32 version).
 *
 * mic:       Microphone input (mono, float32, [-1,1] range, 16kHz)
 * ref:       Far-end reference (mono, float32, [-1,1] range, 16kHz)
 * n_samples: Number of samples in both mic and ref (must be >= 512)
 * out:       Pre-allocated output buffer (n_samples floats)
 *
 * Returns 0 on success, negative on error.
 */
LOCALVQE_API int localvqe_process_f32(uintptr_t ctx,
                                     const float* mic, const float* ref,
                                     int n_samples, float* out);

/**
 * Process audio through the AEC model (int16 PCM version).
 *
 * mic:       Microphone input (mono, int16 PCM, 16kHz)
 * ref:       Far-end reference (mono, int16 PCM, 16kHz)
 * n_samples: Number of samples in both mic and ref (must be >= 512)
 * out:       Pre-allocated output buffer (n_samples int16s)
 *
 * Returns 0 on success, negative on error.
 */
LOCALVQE_API int localvqe_process_s16(uintptr_t ctx,
                                     const int16_t* mic, const int16_t* ref,
                                     int n_samples, int16_t* out);

/**
 * Get the last error message, or empty string if no error.
 * The returned pointer is valid until the next API call on this context.
 */
LOCALVQE_API const char* localvqe_last_error(uintptr_t ctx);

/**
 * Get model sample rate (always 16000 currently).
 */
LOCALVQE_API int localvqe_sample_rate(uintptr_t ctx);

/**
 * Get hop length in samples (256).
 */
LOCALVQE_API int localvqe_hop_length(uintptr_t ctx);

/**
 * Get FFT size (512).
 */
LOCALVQE_API int localvqe_fft_size(uintptr_t ctx);

/**
 * Process a single hop of audio through the AEC model (float32 version).
 *
 * mic:         Microphone input (mono, float32, [-1,1], 16kHz)
 * ref:         Far-end reference (mono, float32, [-1,1], 16kHz)
 * hop_samples: Must equal hop_length (256)
 * out:         Pre-allocated output buffer (hop_samples floats)
 *
 * Returns 0 on success. First call outputs zeros (warmup).
 */
LOCALVQE_API int localvqe_process_frame_f32(uintptr_t ctx,
                                           const float* mic, const float* ref,
                                           int hop_samples, float* out);

/**
 * Process a single hop of audio through the AEC model (int16 PCM version).
 *
 * mic:         Microphone input (mono, int16 PCM, 16kHz)
 * ref:         Far-end reference (mono, int16 PCM, 16kHz)
 * hop_samples: Must equal hop_length (256)
 * out:         Pre-allocated output buffer (hop_samples int16s)
 *
 * Returns 0 on success. First call outputs zeros (warmup).
 */
LOCALVQE_API int localvqe_process_frame_s16(uintptr_t ctx,
                                           const int16_t* mic, const int16_t* ref,
                                           int hop_samples, int16_t* out);

/**
 * Reset streaming state to initial zeros.
 * Call between utterances or when restarting processing.
 */
LOCALVQE_API void localvqe_reset(uintptr_t ctx);

#ifdef __cplusplus
}
#endif

#endif /* LOCALVQE_API_H */
