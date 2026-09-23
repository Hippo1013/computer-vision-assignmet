#!/usr/bin/env bash
# Compile in an isolated system temporary directory; retain only the requested PDF.
set -euo pipefail
cd "$(dirname "$0")"
REPORT_SOURCE=${1:-main.tex}
REPORT_OUTPUT=${2:-report.pdf}
if [[ "$REPORT_SOURCE" != *.tex || ! -f "$REPORT_SOURCE" ]]; then
  echo "Expected an existing .tex source: $REPORT_SOURCE" >&2
  exit 2
fi
command -v latexmk >/dev/null || { echo 'latexmk is required (MacTeX / TeX Live).' >&2; exit 2; }
command -v xelatex >/dev/null || { echo 'XeLaTeX is required (MacTeX / TeX Live).' >&2; exit 2; }
REPORT_BUILD_TMP=$(mktemp -d "${TMPDIR:-/tmp}/qinglan-latex.XXXXXX")
trap 'python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$REPORT_BUILD_TMP"' EXIT
if ! latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -outdir="$REPORT_BUILD_TMP" "$REPORT_SOURCE" > "$REPORT_BUILD_TMP/compile.txt" 2>&1; then
  cat "$REPORT_BUILD_TMP/compile.txt" >&2
  exit 1
fi
REPORT_JOB=$(basename "$REPORT_SOURCE" .tex)
if grep -E 'Overfull \\[hv]box|Missing character:|LaTeX Warning:.*undefined|LaTeX Warning:.*multiply defined' \
    "$REPORT_BUILD_TMP/$REPORT_JOB.log"; then
  echo 'Layout or reference check failed; fix the messages above before publishing.' >&2
  exit 1
fi
cp "$REPORT_BUILD_TMP/$REPORT_JOB.pdf" "$REPORT_OUTPUT"
printf 'Built: %s\n' "$REPORT_OUTPUT"
