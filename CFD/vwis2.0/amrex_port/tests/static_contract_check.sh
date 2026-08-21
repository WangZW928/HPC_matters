#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
port="$root/amrex_port"
required=("$port/CMakeLists.txt" "$port/CMakePresets.json" "$port/amrex_version.lock" "$port/inputs/p1_smoke.in" "$port/inputs/p1_multibox.in" "$port/inputs/p1_contract.in" "$port/inputs/p2_contract.in" "$port/inputs/p2_boundary_face.in" "$port/src/VwisAmrExSolver.H" "$port/src/VwisAmrExSolver.cpp")
for file in "${required[@]}"; do test -f "$file"; done
rg -q 'find_package\(AMReX CONFIG QUIET\)' "$port/CMakeLists.txt"
rg -q 'AMReXConfig.cmake was not found' "$port/CMakeLists.txt"
rg -q 'FieldLocation' "$port/src/VwisAmrExSolver.H"
rg -q 'component_names' "$port/src/VwisAmrExSolver.H"
rg -q 'm_cell_volume' "$port/src/VwisAmrExSolver.H"
rg -q 'm_face_area' "$port/src/VwisAmrExSolver.H"
rg -q 'm_ucont_older' "$port/src/VwisAmrExSolver.H"
rg -q 'amrex::BCRec' "$port/src/VwisAmrExSolver.H"
rg -q 'AMREX_GPU_DEVICE' "$port/src/VwisAmrExSolver.cpp"
rg -q 'payload_written' "$port/src/VwisAmrExSolver.cpp"
rg -q 'run_runtime_contract_checks' "$port/src/VwisAmrExSolver.cpp"
rg -q 'sync_ucat_from_ucont' "$port/src/VwisAmrExSolver.cpp"
rg -q 'sync_ucont_from_ucat' "$port/src/VwisAmrExSolver.cpp"
rg -q 'OverrideSync' "$port/src/VwisAmrExSolver.cpp"
rg -q 'OwnerMask' "$port/src/VwisAmrExSolver.cpp"
rg -q 'sum_unique' "$port/src/VwisAmrExSolver.cpp"
rg -q 'normal velocity times face area' "$port/src/VwisAmrExSolver.cpp"
rg -q 'derived divergence' "$port/src/VwisAmrExSolver.cpp"
rg -q 'constant Ucat/Ucont volume-flux contract failed' "$port/src/VwisAmrExSolver.cpp"
rg -q 'base runtime contract: PASS' "$port/src/VwisAmrExSolver.cpp"
rg -q 'P2-003/004/005: PASS' "$port/src/VwisAmrExSolver.cpp"
if rg -n 'MacProjector|MLMG|PoissonSolver|ComputeRHS|RHSSolver' "$port/src"; then
  echo 'P1 source unexpectedly references a physics solver' >&2
  exit 1
fi
echo 'static P0-P2 AMReX contract check: PASS'
