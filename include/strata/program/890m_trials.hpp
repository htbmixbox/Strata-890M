// Opt-in performance comparisons for a real Radeon 890M ONLY.
// No trial is a default. The 890M name + gfx1150 + 16-CU check occurs before calling this.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::program {
inline bool apply_890m_trial(const char* trial) {
    if (!trial || !*trial) return true;
    auto add = [](const char* key, const char* value) {
        // Explicit user settings always have precedence.
        if (std::getenv(key)) return;
#if defined(_WIN32)
        _putenv_s(key, value);
#else
        setenv(key, value, 0);
#endif
    };
    if (std::strcmp(trial, "wmma") == 0) {
        add("STRATA_HIP_WMMA", "1");
        std::fprintf(stderr, "strata 890M trial: wmma (opt-in, validate outputs and speed)\n");
        return true;
    }
    if (std::strcmp(trial, "fused") == 0) {
        add("STRATA_PF_FUSED", "1");
        std::fprintf(stderr, "strata 890M trial: fused (opt-in, validate outputs and speed)\n");
        return true;
    }
    if (std::strcmp(trial, "gemm") == 0) {
        add("STRATA_PF_GEMM", "1");
        std::fprintf(stderr, "strata 890M trial: gemm (opt-in, validate outputs and speed)\n");
        return true;
    }
    if (std::strcmp(trial, "fused-gemm") == 0) {
        add("STRATA_PF_FUSED", "1");
        add("STRATA_PF_GEMM", "1");
        add("STRATA_PF_SWITCH_MIN_T", "4096");
        std::fprintf(stderr, "strata 890M trial: fused-gemm (opt-in, validate outputs and speed)\n");
        return true;
    }
    if (std::strcmp(trial, "wmma-fused") == 0) {
        add("STRATA_HIP_WMMA", "1");
        add("STRATA_PF_FUSED", "1");
        add("STRATA_PF_SWITCH_MIN_T", "4096");
        std::fprintf(stderr, "strata 890M trial: wmma-fused (opt-in, validate outputs and speed)\n");
        return true;
    }
    if (std::strcmp(trial, "wmma-gemm") == 0) {
        add("STRATA_HIP_WMMA", "1");
        add("STRATA_PF_GEMM", "1");
        add("STRATA_PF_SWITCH_MIN_T", "4096");
        std::fprintf(stderr, "strata 890M trial: wmma-gemm (opt-in, validate outputs and speed)\n");
        return true;
    }
    if (std::strcmp(trial, "wmma-fused-gemm") == 0) {
        add("STRATA_HIP_WMMA", "1");
        add("STRATA_PF_FUSED", "1");
        add("STRATA_PF_GEMM", "1");
        add("STRATA_PF_SWITCH_MIN_T", "4096");
        std::fprintf(stderr, "strata 890M trial: wmma-fused-gemm (opt-in, validate outputs and speed)\n");
        return true;
    }
    if (std::strcmp(trial, "prompt-attn-fast") == 0) {
        add("STRATA_PA_FAST", "1");
        std::fprintf(stderr, "strata 890M trial: prompt-attn-fast (opt-in, validate outputs and speed)\n");
        return true;
    }
    if (std::strcmp(trial, "qsa-select-wmma") == 0) {
        add("STRATA_SELECT_WMMA", "1");
        std::fprintf(stderr, "strata 890M trial: qsa-select-wmma (opt-in, validate outputs and speed)\n");
        return true;
    }
    return false;
}
} // namespace strata::program
