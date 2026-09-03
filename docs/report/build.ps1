# Rebuild the living technical report (Windows / PowerShell).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $here "..\..")
$fig  = Join-Path $here "figures"
New-Item -ItemType Directory -Force -Path $fig | Out-Null

function Copy-Fig($expDir, $target) {
    $src = Join-Path $root "experiments\$expDir\figure.png"
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $fig $target) -Force
        Write-Host "  figure: $expDir  ->  figures/$target"
    } else {
        Write-Warning "missing $src (run its run.py first)"
    }
}

Write-Host "refreshing figures..."
Copy-Fig "03_pid_stabilizes_unstable" "exp03_pid_unstable.png"

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
