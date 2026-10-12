#!/bin/sh
# Radeon 890M (gfx1150) on a big-RAM box: how much of the RAM can the GPU use, and what to set if that is too little.
# Read-only: it changes nothing.  See docs/GFX1150_890M.md.   Usage: tools/hip/gfx1150-memory-check.sh
set -u

ram_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)
[ -n "${ram_kib:-}" ] || { echo "cannot read /proc/meminfo (Linux only)"; exit 1; }
ram_gib=$(( ram_kib / 1048576 ))

gtt=""; vram=""
for d in /sys/class/drm/card*/device; do
    [ -r "$d/mem_info_gtt_total" ] || continue
    # the integrated GPU: the one whose VRAM is the small BIOS carve-out (a discrete card's is its own memory)
    v=$(cat "$d/mem_info_vram_total" 2>/dev/null || echo 0)
    g=$(cat "$d/mem_info_gtt_total" 2>/dev/null || echo 0)
    if [ "$v" -le $(( 8 * 1073741824 )) ] && [ "$g" -gt 0 ]; then gtt=$g; vram=$v; dev=$d; break; fi
done

echo "RAM: ${ram_gib} GiB"
if [ -z "$gtt" ]; then
    echo "no integrated AMD GPU found under /sys/class/drm (is amdgpu loaded, and is this the machine with the 890M?)"
    exit 1
fi
gtt_gib=$(( gtt / 1073741824 ))
vram_mib=$(( vram / 1048576 ))
echo "GPU ($dev): BIOS carve-out ${vram_mib} MiB, GTT pool ${gtt_gib} GiB"

# the pool the GPU may use is capped by ttm.pages_limit (4 KiB pages); the default is about half of the RAM
want_gib=$(( ram_gib - 16 ))
[ "$want_gib" -ge 8 ] || want_gib=8
pages=$(( want_gib * 262144 ))
echo "kernel command line: $(tr ' ' '\n' < /proc/cmdline | grep -E '^(ttm\.|amdgpu\.|amd_iommu)' | tr '\n' ' ')"

if [ "$gtt_gib" -lt "$want_gib" ]; then
    echo
    echo "The GTT pool (${gtt_gib} GiB) is smaller than RAM less 16 GiB (${want_gib} GiB). A model whose experts are all kept"
    echo "in the GTT pool (--mmap-experts --expert-cache N) needs that much room. To raise it, add to the kernel command line"
    echo "(GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub, then update-grub and reboot):"
    echo
    echo "    ttm.pages_limit=${pages} ttm.page_pool_size=${pages}"
    echo
    echo "(${want_gib} GiB in 4 KiB pages.  The measured 890M machine used 16777216 = 64 GiB.)  Leave the rest for the OS, the"
    echo "desktop and the CPU-side buffers: the engine's --expert-cache auto counts MemAvailable less STRATA_UMA_HEADROOM_GIB (6)."
else
    echo "The GTT pool is already RAM less 16 GiB or more: nothing to raise."
fi
if [ "$vram_mib" -gt 2048 ]; then
    echo
    echo "The BIOS carve-out is ${vram_mib} MiB. The engine uses the GTT pool, so a carve-out above ~512 MiB only takes RAM away"
    echo "from the system: the measured 890M machine ran with 512 MiB (UMA frame buffer size, in the BIOS' iGPU settings)."
fi
