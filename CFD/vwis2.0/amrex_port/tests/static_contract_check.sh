#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
port="$root/amrex_port"
required=("$port/CMakeLists.txt" "$port/CMakePresets.json" "$port/amrex_version.lock" "$port/inputs/p1_smoke.in" "$port/inputs/p1_multibox.in" "$port/src/VwisAmrExSolver.H" "$port/src/VwisAmrExSolver.cpp")
for file in "${required[@]}"; do test -f "$file"; done
rg -q 'find_package\(AMReX CONFIG QUIET\)' "$port/CMakeLists.txt"
rg -q 'AMReXConfig.cmake was not found' "$port/CMakeLists.txt"
rg -q 'FieldLocation' "$port/src/VwisAmrExSolver.H"
rg -q 'm_ucont_older' "$port/src/VwisAmrExSolver.H"
rg -q 'amrex::BCRec' "$port/src/VwisAmrExSolver.H"
rg -q 'AMREX_GPU_DEVICE' "$port/src/VwisAmrExSolver.cpp"
rg -q 'payload_written' "$port/src/VwisAmrExSolver.cpp"
if rg -n 'MacProjector|MLMG|PoissonSolver|ComputeRHS|RHSSolver' "$port/src"; then
  echo 'P1 source unexpectedly references a physics solver' >&2
  exit 1
fi
echo 'static P0/P1 AMReX contract check: PASS'
