#!/usr/bin/env bash
# Rebuild the living technical report.
# Run from anywhere; paths are resolved relative to this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
FIG="$HERE/figures"
mkdir -p "$FIG"

# --- 1. refresh figures from the experiments (each writes figure.png next to run.py)
copy_fig() {  # <experiment-dir> <target-name>
  local src="$ROOT/experiments/$1/figure.png"
  if [[ -f "$src" ]]; then
    cp "$src" "$FIG/$2"
    echo "  figure: $1  ->  figures/$2"
  else
    echo "  WARNING: missing $src (run its run.py first)" >&2
  fi
}
echo "refreshing figures..."
copy_fig "03_pid_stabilizes_unstable" "exp03_pid_unstable.png"
# copy_fig "04_lqr_vs_pole_placement_cartpole" "exp04_lqr_cartpole.png"

# --- 2. compile (latexmk drives pdflatex, 2-3 passes for the ToC)
echo "compiling main.tex..."
cd "$HERE"
latexmk -pdf -interaction=nonstopmode -halt-on-error -silent main.tex

# --- 3. publish the PDF under a stable name, clean aux files
cp "$HERE/main.pdf" "$HERE/ai-meets-control-theory.pdf"
latexmk -c -silent main.tex >/dev/null 2>&1 || true
echo "done -> docs/report/ai-meets-control-theory.pdf"
