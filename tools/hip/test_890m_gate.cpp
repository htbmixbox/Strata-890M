#include "strata/program/gfx1150_890m.hpp"
#include <cassert>
int main() {
    using strata::program::is_radeon_890m;
    assert(is_radeon_890m("AMD Radeon(TM) 890M Graphics", "gfx1150", 16));
    assert(!is_radeon_890m("AMD Radeon(TM) 880M Graphics", "gfx1150", 16));
    assert(!is_radeon_890m("AMD Radeon(TM) 890M Graphics", "gfx1151", 16));
    assert(!is_radeon_890m("AMD Radeon(TM) 890M Graphics", "gfx1150", 8));
}
