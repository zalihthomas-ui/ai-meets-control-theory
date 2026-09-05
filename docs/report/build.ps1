# Rebuild the living technical report (Windows / PowerShell).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $here "..\..")
$fig  = Join-Path $here "figures"
New-Item -ItemType Directory -Force -Path $fig | Out-Null

function Copy-Fig($expDir, $sourceFig, $target) {
    $src = Join-Path $root "experiments\$expDir\$sourceFig"
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $fig $target) -Force
        Write-Host "  figure: $expDir\$sourceFig  ->  figures/$target"
    } else {
        Write-Warning "missing $src (run its run.py first)"
    }
}

Write-Host "refreshing figures..."
Copy-Fig "03_pid_stabilizes_unstable" "figure.png" "exp03_pid_unstable.png"
Copy-Fig "04_lqr_vs_pole_placement_cartpole" "figure.png" "exp04_cartpole.png"
Copy-Fig "05_cartpole_basin_of_attraction" "basin_map.png" "exp05_basin_map.png"
Copy-Fig "05_cartpole_basin_of_attraction" "robustness_sweep.png" "exp05_robustness_sweep.png"
Copy-Fig "06_lqg_vs_lqr_measurement_noise" "figure.png" "exp06_lqg.png"
Copy-Fig "07_cartpole_swingup_hybrid" "figure.png" "exp07_figure.png"
Copy-Fig "07_cartpole_swingup_hybrid" "swingup_energy.png" "exp07_swingup_energy.png"
Copy-Fig "08_mpc_vs_lqr_constrained_cartpole" "figure.png" "exp08_figure.png"
Copy-Fig "08_mpc_vs_lqr_constrained_cartpole" "cart_constraint.png" "exp08_cart_constraint.png"
Copy-Fig "09_control_on_identified_model" "figure.png" "exp09_sysid.png"
Copy-Fig "10_planning_learned_vs_true_model" "figure.png" "exp10_learned_mpc.png"
Copy-Fig "11_qlearning_vs_classical" "figure.png" "exp11_rl_pendulum.png"
Copy-Fig "12_shielded_qlearning" "figure.png" "exp12_shielded.png"
Copy-Fig "14_quadrotor_figure8_tracking" "figure.png" "exp14_quadrotor.png"
Copy-Fig "15_quadrotor_ekf_output_feedback" "figure.png" "exp15_quadrotor_ekf.png"
Copy-Fig "16_ekf_vs_ukf" "figure.png" "exp16_ekf_vs_ukf.png"
Copy-Fig "17_adaptive_vs_fixed_changing_plant" "figure.png" "exp17_adaptive.png"
Copy-Fig "18_rl_zoo_vs_lqr" "figure.png" "exp18_rl_zoo.png"
Copy-Fig "19_icc_leaderboard" "figure.png" "exp19_icc_leaderboard.png"
Copy-Fig "20_quadrotor_obstacle_nmpc" "figure.png" "exp20_obstacle_nmpc.png"
Copy-Fig "21_grand_capstone_bakeoff" "figure.png" "exp21_grand_bakeoff.png"
Copy-Fig "22_diffdrive_path_following" "tracking.png" "exp22_diffdrive.png"
Copy-Fig "23_twolink_arm_tracking" "tracking.png" "exp23_twolink_tracking.png"
Copy-Fig "23_twolink_arm_tracking" "payload.png" "exp23_twolink_payload.png"
Copy-Fig "24_ilqr_vs_sampling_mpc" "figure.png" "exp24_ilqr_vs_sampling_mpc.png"
Copy-Fig "25_diffdrive_moving_obstacle" "figure.png" "exp25_diffdrive_moving_obstacle.png"
Copy-Fig "26_harder_reference_paths" "figure.png" "exp26_harder_reference_paths.png"
Copy-Fig "27_bicycle_double_lane_change" "figure.png" "exp27_bicycle_double_lane_change.png"
Copy-Fig "28_furuta_pendulum_control" "furuta_benchmark.png" "exp28_furuta_benchmark.png"
Copy-Fig "28_furuta_pendulum_control" "furuta_swingup.png" "exp28_furuta_swingup.png"
Copy-Fig "29_dagger_vs_bc_lane_change" "figure.png" "exp29_dagger_recovery.png"
Copy-Fig "30_two_tank_level_control" "two_tank_benchmark.png" "exp30_two_tank_benchmark.png"
Copy-Fig "30_two_tank_level_control" "two_tank_setpoint.png" "exp30_two_tank_setpoint.png"
Copy-Fig "31_sac_vs_ppo_sample_efficiency" "figure.png" "exp31_sac_vs_ppo.png"


Write-Host "compiling main.tex..."
Push-Location $here
try {
    # Run pdflatex twice to resolve references and table of contents
    pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
    pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
    Copy-Item (Join-Path $here "main.pdf") (Join-Path $here "ai-meets-control-theory.pdf") -Force
    # Clean up auxiliary files
    Get-ChildItem -Path $here -Include *.aux, *.log, *.out, *.toc -File | Remove-Item -Force
} finally {
    Pop-Location
}
Write-Host "done -> docs/report/ai-meets-control-theory.pdf"
