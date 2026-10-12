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
    return false;
}
} // namespace strata::program
