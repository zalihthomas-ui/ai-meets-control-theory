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
