// A 4-row verify window = three sequential probabilistic MTP proposals.
// These are NOT three parallel branches; the existing exact rejection sampler
// remains the only authority for accepted target-distributed tokens.
#pragma once
#include <string>

namespace strata::program {
inline std::string validate_mtp_prob3(const std::string& mtp, int spec, int mtp_max_t,
                                      int lookup_chain, double min_p, int batch, bool batch_mtp) {
    if (mtp.empty()) return "--890m-mtp-prob3 requires --mtp PATH";
    if (spec != 0 && spec != 4) return "--890m-mtp-prob3 conflicts with --spec other than 4";
    if (mtp_max_t != 0 && mtp_max_t != 4) return "--890m-mtp-prob3 conflicts with --mtp-max-t other than 4";
    if (lookup_chain != 0) return "--890m-mtp-prob3 requires --lookup-chain 0";
    if (min_p != 0.0) return "--890m-mtp-prob3 requires --spec-min-p 0 (all three proposals)";
    if (batch > 0 || batch_mtp) return "--890m-mtp-prob3 requires unbatched MTP verification";
    return {};
}
} // namespace strata::program
