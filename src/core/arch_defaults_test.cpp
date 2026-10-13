// src/core/arch_defaults_test.cpp - the per-architecture default table (no GPU needed).
#include "strata/core/arch_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

int main() {
    using namespace strata::core;
    unset("STRATA_GFX1151_DEFAULTS");
    unset("STRATA_GFX1150_DEFAULTS");
    unset("STRATA_GFX1150_FAST");
    // other architectures (and CUDA, which passes an empty name): nothing
    CHECK(arch_default_env("gfx1100").empty());
    CHECK(arch_default_env("gfx1152").empty());
    CHECK(arch_default_env("gfx11500").empty());   // not a prefix match
    CHECK(arch_default_env("gfx11510").empty());   // not a prefix match
    CHECK(arch_default_env("").empty());
    CHECK(arch_default_env(nullptr).empty());
    // gfx1151, plain and with its feature suffix: the table, and no bit-changing switch in it
    for (const char* a : {"gfx1151", "gfx1151:sramecc-:xnack-"}) {
        const auto t = arch_default_env(a);
        CHECK(t.size() >= 10);
        for (const auto& kv : t) {
            CHECK(kv.first.rfind("STRATA_", 0) == 0);
            for (const char* bits : {"STRATA_PF_GEMM", "STRATA_PF_FUSED", "STRATA_HC_UPMIX", "STRATA_PA_FAST", "STRATA_HIP_WMMA",
                                     "STRATA_SELECT_WMMA", "STRATA_HC_Q8", "STRATA_PF_HCDOWN", "STRATA_WMMA_GEMM"})
                CHECK(kv.first != bits);
        }
    }
    // gfx1150 (Radeon 890M): the gfx1151 table less STRATA_HCD_EXACT, no bit-changing switch by default
    {
        const auto t151 = arch_default_env("gfx1151");
        for (const char* a : {"gfx1150", "gfx1150:sramecc-:xnack-"}) {
            const auto t = arch_default_env(a);
            CHECK(t.size() == t151.size() - 1);
            for (const auto& kv : t) {
                CHECK(kv.first != "STRATA_HCD_EXACT");
                for (const char* bits : {"STRATA_PF_GEMM", "STRATA_PF_FUSED", "STRATA_HC_UPMIX", "STRATA_PA_FAST", "STRATA_HIP_WMMA",
                                         "STRATA_SELECT_WMMA", "STRATA_HC_Q8", "STRATA_PF_HCDOWN", "STRATA_WMMA_GEMM",
                                         "STRATA_PF_SWITCH_MIN_T"})
                    CHECK(kv.first != bits);
            }
        }
        // the table is the gfx1151 one: every other switch of it is there
        for (const auto& kv : t151) {
            if (kv.first == "STRATA_HCD_EXACT") continue;
            bool found = false;
            for (const auto& k2 : arch_default_env("gfx1150")) found = found || (k2.first == kv.first && k2.second == kv.second);
            CHECK(found);
        }
        // the opt-in rounding-level groups
        put("STRATA_GFX1150_FAST", "1");
        auto f1 = arch_default_env("gfx1150");
        CHECK(f1.size() == t151.size());   // -HCD_EXACT +HIP_WMMA
        bool wmma = false, fused = false;
        for (const auto& kv : f1) { wmma = wmma || kv.first == "STRATA_HIP_WMMA"; fused = fused || kv.first == "STRATA_PF_FUSED"; }
        CHECK(wmma && !fused);
        put("STRATA_GFX1150_FAST", "2");
        auto f2 = arch_default_env("gfx1150");
        CHECK(f2.size() == t151.size() + 3);
        put("STRATA_GFX1150_FAST", "0");
        CHECK(arch_default_env("gfx1150").size() == t151.size() - 1);
        // FAST never touches another architecture
        put("STRATA_GFX1150_FAST", "2");
        CHECK(arch_default_env("gfx1151").size() == t151.size());
        unset("STRATA_GFX1150_FAST");
        // the opt-out
        put("STRATA_GFX1150_DEFAULTS", "0");
        CHECK(arch_default_env("gfx1150").empty());
        CHECK(apply_arch_defaults("gfx1150").empty());
        unset("STRATA_GFX1150_DEFAULTS");
        // a user's setting wins; a named hipBLASLt table is kept
        put("STRATA_TSUM", "0");
        put("STRATA_HIPBLASLT_TUNING", "/nonexistent/table.txt");
        unset("STRATA_HIPBLASLT_HEURISTIC");
        const auto set50 = apply_arch_defaults("gfx1150");
        // the hipBLASLt heuristic is on for gfx1150 where the user named nothing ...
        CHECK(std::getenv("STRATA_HIPBLASLT_HEURISTIC") != nullptr && std::string(std::getenv("STRATA_HIPBLASLT_HEURISTIC")) == "1");
        bool heur_named = false;
        for (const auto& n : set50) heur_named = heur_named || n == "STRATA_HIPBLASLT_HEURISTIC";
        CHECK(heur_named);
        // ... and never overrides the user's own (0 included), nor touches another architecture
        put("STRATA_HIPBLASLT_HEURISTIC", "0");
        for (const auto& n : apply_arch_defaults("gfx1150")) CHECK(n != "STRATA_HIPBLASLT_HEURISTIC");
        CHECK(std::string(std::getenv("STRATA_HIPBLASLT_HEURISTIC")) == "0");
        unset("STRATA_HIPBLASLT_HEURISTIC");
        for (const auto& n : apply_arch_defaults("gfx1151")) CHECK(n != "STRATA_HIPBLASLT_HEURISTIC");
        CHECK(std::getenv("STRATA_HIPBLASLT_HEURISTIC") == nullptr);
        CHECK(std::string(std::getenv("STRATA_TSUM")) == "0");
        CHECK(std::string(std::getenv("STRATA_HIPBLASLT_TUNING")) == "/nonexistent/table.txt");
        CHECK(!set50.empty());
        for (const auto& n : set50) CHECK(n != "STRATA_TSUM" && n != "STRATA_HIPBLASLT_TUNING");
        unset("STRATA_TSUM");
        unset("STRATA_HIPBLASLT_TUNING");
        for (const auto& kv : t151) unset(kv.first.c_str());
    }
    // gfx12: the five prompt switches only, no decode switch, no bit-changing one
    unset("STRATA_GFX12_DEFAULTS");
    for (const char* a : {"gfx1200", "gfx1201", "gfx1201:sramecc-:xnack-"}) {
        const auto t = arch_default_env(a);
        CHECK(t.size() == 5);
        for (const auto& kv : t) CHECK(kv.first != "STRATA_HCD_EXACT");
        for (const auto& kv : t) {
            CHECK(kv.first != "STRATA_Q8_PACKED" && kv.first != "STRATA_SH_STREAM" && kv.first != "STRATA_PF_GEMM" &&
                  kv.first != "STRATA_PF_FUSED");
        }
    }
    CHECK(arch_default_env("gfx12010").empty());
    put("STRATA_GFX12_DEFAULTS", "0");
    CHECK(arch_default_env("gfx1201").empty());
    unset("STRATA_GFX12_DEFAULTS");
    // a user's setting wins, whatever it is; the rest are set
    put("STRATA_TSUM", "0");
    unset("STRATA_QFUSE");
    const auto set = apply_arch_defaults("gfx1151");
    CHECK(std::getenv("STRATA_TSUM") != nullptr && std::string(std::getenv("STRATA_TSUM")) == "0");
    CHECK(std::getenv("STRATA_QFUSE") != nullptr && std::string(std::getenv("STRATA_QFUSE")) == "1");
    for (const auto& n : set) CHECK(n != "STRATA_TSUM");
    // the opt-out
    put("STRATA_GFX1151_DEFAULTS", "0");
    CHECK(arch_default_env("gfx1151").empty());
    CHECK(apply_arch_defaults("gfx1151").empty());
    std::printf(fails ? "arch_defaults_test: %d FAILED\n" : "arch_defaults_test: OK\n", fails);
    return fails ? 1 : 0;
}
