#!/usr/bin/env bash
set -euo pipefail

case_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$case_dir/../.." && pwd)
exe=${VWIS_AMREX_EXE:-"$project_dir/build/amrex_port_p8/vwis_amrex_skeleton"}

if [[ ! -x "$exe" ]]; then
  echo "AMReX executable is not available: $exe" >&2
  exit 1
fi

mkdir -p "$case_dir/raw" "$case_dir/figures"
mkdir -p /tmp/vwis-lid-matplotlib-cache
rm -f "$case_dir/raw/summary.json" "$case_dir/raw/final_field.csv" \
      "$case_dir/raw/centerlines.csv" "$case_dir/raw/history.csv" \
      "$case_dir/raw/solver.log" "$case_dir/raw/command.txt"
printf '%q %q\n' "$exe" "$case_dir/inputs.in" > "$case_dir/raw/command.txt"
(
  cd "$case_dir"
  "$exe" inputs.in 2>&1 | tee raw/solver.log
)
MPLCONFIGDIR=/tmp/vwis-lid-matplotlib-cache python3 "$case_dir/plot_results.py"
