// src/core/arch_defaults.cpp - see include/strata/core/arch_defaults.hpp.
#include "strata/core/arch_defaults.hpp"

#include "strata/kernels/gfx_arch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

namespace strata::core {

namespace {

bool env_is(const char* name, char c) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == c;
}

// The exact-by-construction switches measured on gfx1151 (see arch_default_env).
std::vector<std::pair<std::string, std::string>> gfx1151_table() {
    return {
        // prompt
        {"STRATA_GDN_HEAD", "1"},        // the GDN recurrence, four lanes a column + the grid-stride norm (S23)
        {"STRATA_GDN_PP", "2"},          //   two columns per lane: 64K prompt +2.9% (stream A, round 2)
        {"STRATA_GDN_CONVL2", "1"},      // conv + q/k L2 norm in one kernel: 64K -0.3 s (stream A)
        {"STRATA_GDN_NOY", "1"},         // no dead FP32 store of the norm (stream A)
        {"STRATA_CVEC_FUSE", "1"},       // a steered layer's write, control vector and next norm in one pass (S23)
        {"STRATA_HCD_EXACT", "1"},       // the HC down GEMM in hipBLASLt solution 1176's k order: 64K +1.8% (stream S)
        {"STRATA_PF_PAD", "1"},          // padded GEMM row strides (acts with STRATA_PF_GEMM; inert without it) (S23)
        // decode
        {"STRATA_Q8_PACKED", "1"},       // the packed Q8_0 decode layout: +2.3% (S26)
        {"STRATA_Q6_PACKED", "1"},       // the packed Q6_K heads (S26)
        {"STRATA_MMVF_ROWS", "1"},       // bf16 multi-row GEMV, 4 rows per block (S25)
        {"STRATA_ATTN_LANECELL", "1"},   // decode attention scores, one cell per thread (S25)
        {"STRATA_EXPERT_V2", "1"},       // the grouped decode experts, IQ3_S gate/up + IQ4_NL down (S26)
        {"STRATA_TSUM", "1"},            // several warp sums as one transposed butterfly (S26)
        {"STRATA_LFUSE", "1"},           // fewer launches around the shared expert and the KV append (S26)
        {"STRATA_GDN_SPLIT", "1"},       // the GDN step over 4 blocks per head (S25/S26)
        {"STRATA_QFUSE", "1"},           // activation q8_1 images written by their producers (S26)
        {"STRATA_PLE_BATCH", "1"},       // the verify window's PLE key / value projections at once (S25)
        {"STRATA_SH_STREAM", "1"},       // the shared expert on its own stream: decode +1.8% / +6.7% (UD-Q4_K_XL) (140-m)
    };
}

// The file name of the gfx1150 hipBLASLt table shipped in tools/hip (ROCm 7.14.1, hipBLASLt version 100401).
constexpr const char* kGfx1150Table = "gfx1150-hipblaslt-100401.txt";

// Where the table is looked for: tools/hip/ under the working directory and under the executable's directory and its
// parents (a build tree is usually <repo>/build-x/strata).  Empty when it is nowhere.
std::string find_gfx1150_table() {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<fs::path> roots;
    roots.push_back(fs::current_path(ec));
#if defined(__linux__)
    {
        const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
        if (!ec) {
            fs::path d = exe.parent_path();
            for (int up = 0; up < 5 && !d.empty(); ++up) {
                roots.push_back(d);
                const fs::path parent = d.parent_path();
                if (parent == d) break;
                d = parent;
            }
        }
    }
#endif
    for (const fs::path& r : roots) {
        if (r.empty()) continue;
        const fs::path cand = r / "tools" / "hip" / kGfx1150Table;
        std::error_code e2;
        if (fs::is_regular_file(cand, e2)) return cand.string();
    }
    return {};
}

}  // namespace

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch) {
    std::vector<std::pair<std::string, std::string>> t;
    if (strata::kernels::gfx_arch_is_gfx1150(gcn_arch)) {
        // gfx1150 (Strix Point, Radeon 890M / 880M, 16 CUs): the gfx1151 table less STRATA_HCD_EXACT.  A community 890M
        // (Ryzen AI 9 HX PRO 370, 96 GB DDR5-5600, bench/results/2026-10-07-community-gfx1150) ran those 17 switches
        // with the same greedy answers with and without them (also the 13,969-token prompt), decode 14.9-15.0 -> 15.2-15.3
        // tok/s (+2%) and prompts +2.1..+6.0%.  STRATA_HCD_EXACT stays out: it copies gfx1151's hipBLASLt solutions
        // 1176 / 1177 and the gfx1150 table picks others (539-555 for the bf16 rows), so it would do nothing and say so.
        if (env_is("STRATA_GFX1150_DEFAULTS", '0')) return t;
        for (auto& kv : gfx1151_table())
            if (kv.first != "STRATA_HCD_EXACT") t.push_back(std::move(kv));
        // Level 1 experiment for Radeon 890M (gfx1150) only: enable HIP WMMA prefill
        // unless explicitly disabled with STRATA_GFX1150_FAST=0. This changes floating-point
        // rounding, so validate output quality and correctness as well as speed.
        // Existing single-run evidence (table + exact switches, ~1K / 3.6K / 7K prompt):
        // baseline 153 / 217 / 230 tok/s; STRATA_HIP_WMMA=1: 160 / 250 / 277 tok/s.
        // Keep the fused/GEMM experiments separate; they have not been measured together.
        const char* fast = std::getenv("STRATA_GFX1150_FAST");
        if (!(fast != nullptr && fast[0] == '0')) {
            t.push_back({"STRATA_HIP_WMMA", "1"});
            if (fast != nullptr && fast[0] == '2') {
                t.push_back({"STRATA_PF_FUSED", "1"});
                t.push_back({"STRATA_PF_GEMM", "1"});
                t.push_back({"STRATA_PF_SWITCH_MIN_T", "4096"});
            }
        }
        return t;
    }
    if (strata::kernels::gfx_arch_is_gfx12(gcn_arch)) {
        // gfx1200 / gfx1201 (RDNA4): only the PROMPT half of the gfx1151 table.  Reported exact (identical answers) on an R9700
        // in #1478 and re-checked bitwise here (notes-142-amd.md); the decode half measured slower on that card, so it stays out.
        // STRATA_HCD_EXACT is left out: it needs the gfx1151 hipBLASLt tuning table, finds none on gfx12, says so on every start
        // and does nothing (measured: with and without it identical, 0.0 to +0.1%).
        // STRATA_GDN_CONVL2 is AMD-only on purpose: it is not bitwise on CUDA 13.3 sm_120 (#1522), and this table is never
        // consulted by a CUDA build.
        const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
        if (off12 != nullptr && off12[0] == '0') return t;
        t = {
            {"STRATA_GDN_HEAD", "1"},
            {"STRATA_GDN_PP", "2"},
            {"STRATA_GDN_CONVL2", "1"},
            {"STRATA_GDN_NOY", "1"},
            {"STRATA_CVEC_FUSE", "1"},
        };
        return t;
    }
    if (!strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    // Exact (bitwise) on Aurora, each with its measured gain in aurora_s23.md (the table is gfx1151_table() above):
    t = gfx1151_table();
    return t;
}

std::vector<std::string> apply_arch_defaults(const char* gcn_arch) {
    std::vector<std::string> set;
    for (const auto& kv : arch_default_env(gcn_arch)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
    }
    const bool g12 = strata::kernels::gfx_arch_is_gfx12(gcn_arch);
    const bool g50 = strata::kernels::gfx_arch_is_gfx1150(gcn_arch);
    if (!set.empty()) {
        std::fprintf(stderr, "strata: %s: %zu speed switches on by default (%s=0 turns "
                             "them off; a switch you set is kept): ",
                     g12 ? "gfx12 (RDNA4)" : g50 ? "gfx1150 (Strix Point, Radeon 890M)" : "gfx1151 (Strix Halo)", set.size(),
                     g12 ? "STRATA_GFX12_DEFAULTS" : g50 ? "STRATA_GFX1150_DEFAULTS" : "STRATA_GFX1151_DEFAULTS");
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    // gfx1150: the hipBLASLt table for the prompt's dense GEMMs, when the user has not named one.  The engine checks the
    // header (architecture and hipBLASLt version) and says so and keeps hipBLASEx if it does not fit this ROCm.
    if (g50 && !env_is("STRATA_GFX1150_DEFAULTS", '0') && std::getenv("STRATA_HIPBLASLT_TUNING") == nullptr) {
        const std::string table = find_gfx1150_table();
        if (!table.empty()) {
#if defined(_WIN32)
            _putenv_s("STRATA_HIPBLASLT_TUNING", table.c_str());
#else
            setenv("STRATA_HIPBLASLT_TUNING", table.c_str(), 0);
#endif
            std::fprintf(stderr, "strata: gfx1150: STRATA_HIPBLASLT_TUNING=%s (found next to the engine; "
                                 "STRATA_GFX1150_DEFAULTS=0 or your own STRATA_HIPBLASLT_TUNING replaces it)\n", table.c_str());
            set.push_back("STRATA_HIPBLASLT_TUNING");
        } else {
            std::fprintf(stderr, "strata: gfx1150: tools/hip/%s not found near the engine or the working directory: the prompt's "
                                 "dense GEMMs run on plain hipBLAS (1.5-1.7x slower reads); set STRATA_HIPBLASLT_TUNING to it\n",
                         kGfx1150Table);
        }
    }
    return set;
}

}  // namespace strata::core
