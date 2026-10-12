// include/strata/core/arch_defaults.hpp - the per-architecture defaults of switches that are exact everywhere.
//
// Every gfx1151 (Strix Halo, RDNA3.5, unified memory) switch in the table was measured on Aurora (experiments/s23 .. s26,
// the c-best3-v4.2 config) with BYTE-IDENTICAL output against the same engine with the switch off: the same token ids at
// 4K / 32K / 64K and on four models, and a bitwise harness per kernel.  They are therefore on by default on that
// architecture and nowhere else: a CUDA build and every other HIP architecture run the code they always ran.
// Switches that change bits (STRATA_PF_GEMM, STRATA_PF_FUSED, STRATA_HC_UPMIX, STRATA_PA_FAST, STRATA_HIP_WMMA,
// STRATA_SELECT_WMMA, STRATA_HC_Q8: rounding-level, KL-gated) stay opt-in; docs/STRIX_HALO.md lists them.
//
// gfx1150 (Strix Point, Radeon 890M / 880M, 16 CUs) gets the gfx1151 table less STRATA_HCD_EXACT (it copies gfx1151's
// hipBLASLt solutions, and the gfx1150 table picks others): a community 890M ran those 17 switches with identical greedy
// answers and decode +2%, prompts +2..6% (bench/results/2026-10-07-community-gfx1150).  On gfx1150 apply_arch_defaults also
// points STRATA_HIPBLASLT_TUNING at tools/hip/gfx1150-hipblaslt-100401.txt when it finds that file and the variable is
// unset (prompts 1.5-1.7x faster on that machine; the engine still refuses the table for another hipBLASLt version).
// STRATA_GFX1150_FAST=1 / 2 add rounding-level (bit-changing) prompt switches on gfx1150, opt-in only.
//
// A switch the user set (to anything) is never overridden; STRATA_GFX1151_DEFAULTS=0 (gfx1150: STRATA_GFX1150_DEFAULTS=0)
// turns the whole table off.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The environment settings that are the defaults of `gcn_arch` (hipDeviceProp_t::gcnArchName; "gfx1151" or
/// "gfx1151:sramecc-:xnack-"): empty for an architecture without a table.
std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch);

/// Sets `arch_default_env(gcn_arch)` in this process's environment where the user has not set the variable, prints one
/// line to stderr when it set anything, and returns the names it set.  Call once, before any engine code reads its
/// switches (they are read lazily, at first use).
std::vector<std::string> apply_arch_defaults(const char* gcn_arch);

}  // namespace strata::core
