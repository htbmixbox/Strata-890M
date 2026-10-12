#include "strata/program/mtp_prob3_890m.hpp"
#include "strata/core/coupled_draft.hpp"
#include <cassert>
#include <string>
int main() {
    using strata::program::validate_mtp_prob3;
    assert(validate_mtp_prob3("mtp", 0, 0, 0, 0, 0, false).empty());
    assert(validate_mtp_prob3("mtp", 4, 4, 0, 0, 0, false).empty());
    assert(!validate_mtp_prob3("", 4, 4, 0, 0, 0, false).empty());
    assert(!validate_mtp_prob3("mtp", 5, 4, 0, 0, 0, false).empty());
    assert(!validate_mtp_prob3("mtp", 4, 4, 1, 0, 0, false).empty());
    assert(!validate_mtp_prob3("mtp", 4, 4, 0, .2, 0, false).empty());
    assert(!validate_mtp_prob3("mtp", 4, 4, 0, 0, 2, false).empty());
    strata::core::set_spec_prob(true);
    assert(strata::core::spec_prob_env() && strata::core::coupled_draft_env());
    strata::core::set_spec_prob(false);
    assert(!strata::core::spec_prob_env());
}
