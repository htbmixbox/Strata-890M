// Radeon 890M-only gating for opt-in experiments. gfx1150 also names other
// products; neither HIP's architecture string nor the CU count alone suffices.
#pragma once
#include <cstring>

namespace strata::program {
inline bool is_radeon_890m(const char* name, const char* arch, int compute_units) {
    return name && arch && std::strstr(name, "890M") != nullptr &&
           std::strncmp(arch, "gfx1150", 7) == 0 && compute_units == 16;
}
} // namespace strata::program
