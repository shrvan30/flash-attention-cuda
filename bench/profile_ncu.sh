#!/usr/bin/env bash
#
# Nsight Compute capture for the v2 kernels (T1.9).
#
# This script is NOT runnable on a machine whose driver restricts performance
# counters to administrators — it will stop with instructions. Run it on a
# counter-capable RTX 3090 (same GPU model as the published benchmarks, so the
# profiles line up with them).
#
#   bash bench/profile_ncu.sh                 # capture, then extract
#   bash bench/profile_ncu.sh --extract-only  # re-extract from existing reports
#
# Captures `ncu --set full` on:
#   prefill  N=2048, causal, B=4, H=12
#   decode   S=1024, B=1, H=12   (both split and merge kernels)
#
# Writes docs/profiles/ncu_prefill.ncu-rep, docs/profiles/ncu_decode.ncu-rep and
# extracts SM SOL %, memory SOL %, achieved occupancy, DRAM throughput and
# shared-memory bank conflicts into docs/profiles/summary.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE_DIR="${REPO_ROOT}/docs/profiles"
WORKLOAD="${SCRIPT_DIR}/profile_workload.py"
SUMMARY="${PROFILE_DIR}/summary.md"

PYTHON="${PYTHON:-python3}"
NCU="${NCU:-ncu}"

EXTRACT_ONLY=0
if [[ "${1:-}" == "--extract-only" ]]; then
  EXTRACT_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--extract-only]" >&2
  exit 2
fi

mkdir -p "${PROFILE_DIR}"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

if ! command -v "${NCU}" > /dev/null 2>&1; then
  echo "error: ${NCU} not found on PATH." >&2
  echo "Nsight Compute ships with the CUDA toolkit; try /usr/local/cuda/bin/ncu" >&2
  echo "or set NCU=/path/to/ncu." >&2
  exit 1
fi

counter_help() {
  cat >&2 <<'EOF'

This machine denies access to GPU performance counters, so `ncu` cannot run.
On a box where you control the host kernel module, as root:

    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
        > /etc/modprobe.d/nvidia-prof.conf
    update-initramfs -u
    reboot

Then re-run this script. On a container-only instance the parameter cannot be
changed from inside; use a VM-type rental instead.
EOF
}

if [[ ${EXTRACT_ONLY} -eq 0 ]]; then
  if ! "${NCU}" --version > /dev/null 2>&1; then
    echo "error: ${NCU} is present but failed to run." >&2
    exit 1
  fi

  # Cheapest possible probe: profiling permission is checked before anything else.
  if ! "${NCU}" --metrics sm__cycles_elapsed.avg --target-processes all \
      "${PYTHON}" -c "import torch; torch.zeros(1, device='cuda')" \
      > /tmp/ncu_probe.log 2>&1; then
    if grep -q "ERR_NVGPUCTRPERM" /tmp/ncu_probe.log; then
      echo "error: ERR_NVGPUCTRPERM — no access to GPU performance counters." >&2
      counter_help
    else
      echo "error: the ncu probe failed; see /tmp/ncu_probe.log" >&2
      tail -20 /tmp/ncu_probe.log >&2
    fi
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

capture() {
  local name="$1" kernel_regex="$2" launch_count="$3"
  echo ">>> capturing ${name}"
  # The workload warms up 5 times before its timed loop, so skipping 5 launches
  # of the matched kernel lands on a steady-state one.
  "${NCU}" --set full \
    --kernel-name "regex:${kernel_regex}" \
    --launch-skip 5 \
    --launch-count "${launch_count}" \
    --force-overwrite \
    --export "${PROFILE_DIR}/ncu_${name}" \
    "${PYTHON}" "${WORKLOAD}" "${name}"
}

if [[ ${EXTRACT_ONLY} -eq 0 ]]; then
  capture prefill "prefill_kernel" 1
  capture decode "decode_(split|merge)_kernel" 2
fi

for name in prefill decode; do
  if [[ ! -f "${PROFILE_DIR}/ncu_${name}.ncu-rep" ]]; then
    echo "error: ${PROFILE_DIR}/ncu_${name}.ncu-rep is missing." >&2
    echo "Run without --extract-only to capture it first." >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

echo ">>> extracting metrics into ${SUMMARY}"

for name in prefill decode; do
  "${NCU}" --import "${PROFILE_DIR}/ncu_${name}.ncu-rep" --csv --page raw \
    > "${PROFILE_DIR}/.ncu_${name}.csv"
done

"${PYTHON}" "${SCRIPT_DIR}/ncu_extract.py" \
  --prefill "${PROFILE_DIR}/.ncu_prefill.csv" \
  --decode "${PROFILE_DIR}/.ncu_decode.csv" \
  --out "${SUMMARY}"

rm -f "${PROFILE_DIR}/.ncu_prefill.csv" "${PROFILE_DIR}/.ncu_decode.csv"

echo
echo "done:"
echo "  ${PROFILE_DIR}/ncu_prefill.ncu-rep"
echo "  ${PROFILE_DIR}/ncu_decode.ncu-rep"
echo "  ${SUMMARY}"
echo
echo "Next, per T1.9: add the measured-versus-theoretical occupancy comparison"
echo "against docs/profiles/analysis.md, then commit and push."
