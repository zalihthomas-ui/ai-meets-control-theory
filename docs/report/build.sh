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
copy_fig "10_planning_learned_vs_true_model" "figure.png" "exp10_learned_mpc.png"
copy_fig "11_qlearning_vs_classical" "figure.png" "exp11_rl_pendulum.png"
copy_fig "12_shielded_qlearning" "figure.png" "exp12_shielded.png"
copy_fig "14_quadrotor_figure8_tracking" "figure.png" "exp14_quadrotor.png"
copy_fig "15_quadrotor_ekf_output_feedback" "figure.png" "exp15_quadrotor_ekf.png"
copy_fig "16_ekf_vs_ukf" "figure.png" "exp16_ekf_vs_ukf.png"
copy_fig "17_adaptive_vs_fixed_changing_plant" "figure.png" "exp17_adaptive.png"
copy_fig "18_rl_zoo_vs_lqr" "figure.png" "exp18_rl_zoo.png"
copy_fig "19_icc_leaderboard" "figure.png" "exp19_icc_leaderboard.png"
copy_fig "20_quadrotor_obstacle_nmpc" "figure.png" "exp20_obstacle_nmpc.png"
copy_fig "21_grand_capstone_bakeoff" "figure.png" "exp21_grand_bakeoff.png"

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
