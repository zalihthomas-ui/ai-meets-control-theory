#!/usr/bin/env bash
# Rebuild the living technical report.
# Run from anywhere; paths are resolved relative to this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
FIG="$HERE/figures"
mkdir -p "$FIG"

# --- 1. refresh figures from the experiments (each writes figure.png next to run.py)
copy_fig() {  # <experiment-dir> <source-fig> <target-name>
  local src="$ROOT/experiments/$1/$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$FIG/$3"
    echo "  figure: $1/$2  ->  figures/$3"
  else
    echo "  WARNING: missing $src (run its run.py first)" >&2
  fi
}
echo "refreshing figures..."
copy_fig "03_pid_stabilizes_unstable" "figure.png" "exp03_pid_unstable.png"
copy_fig "04_lqr_vs_pole_placement_cartpole" "figure.png" "exp04_cartpole.png"
copy_fig "05_cartpole_basin_of_attraction" "basin_map.png" "exp05_basin_map.png"
copy_fig "05_cartpole_basin_of_attraction" "robustness_sweep.png" "exp05_robustness_sweep.png"
copy_fig "06_lqg_vs_lqr_measurement_noise" "figure.png" "exp06_lqg.png"
copy_fig "07_cartpole_swingup_hybrid" "figure.png" "exp07_figure.png"
copy_fig "07_cartpole_swingup_hybrid" "swingup_energy.png" "exp07_swingup_energy.png"
copy_fig "08_mpc_vs_lqr_constrained_cartpole" "figure.png" "exp08_figure.png"
copy_fig "08_mpc_vs_lqr_constrained_cartpole" "cart_constraint.png" "exp08_cart_constraint.png"
copy_fig "09_control_on_identified_model" "figure.png" "exp09_sysid.png"

# --- 2. compile (latexmk drives pdflatex, 2-3 passes for the ToC)
echo "compiling main.tex..."
cd "$HERE"
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -silent main.tex
  cp "$HERE/main.pdf" "$HERE/ai-meets-control-theory.pdf"
  latexmk -c -silent main.tex >/dev/null 2>&1 || true
else
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
  cp "$HERE/main.pdf" "$HERE/ai-meets-control-theory.pdf"
  rm -f *.aux *.log *.out *.toc
fi
echo "done -> docs/report/ai-meets-control-theory.pdf"
