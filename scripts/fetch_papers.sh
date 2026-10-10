#!/usr/bin/env bash
# Download every paper in papers.tsv into papers/<group>/<arxiv_id>.pdf.
# Polite to arXiv: one request every 3 seconds, skips files already present.
# Usage: scripts/fetch_papers.sh [max_priority]   (e.g. 1 = only must-reads)
set -euo pipefail
cd "$(dirname "$0")/.."
MAXP="${1:-3}"
mkdir -p papers

tail -n +2 papers.tsv | while IFS=$'\t' read -r id year group prio title; do
  [ "$prio" -le "$MAXP" ] || continue
  dir="papers/$group"; out="$dir/$id.pdf"
  mkdir -p "$dir"
  if [ -s "$out" ]; then echo "have  $id  $title"; continue; fi
  echo "fetch $id  $title"
  curl -sSL --fail -A "vibe-husbandry-reading-list/0.1" -o "$out" "https://arxiv.org/pdf/$id" \
    || { echo "  FAILED $id"; rm -f "$out"; }
  sleep 3
done

# Not on arXiv: NEAT (Stanley & Miikkulainen 2002), origin of the "competing conventions" fix.
neat="papers/06-evolution-populations/stanley-miikkulainen-2002-NEAT.pdf"
if [ ! -s "$neat" ]; then
  curl -sSL --fail -o "$neat" "https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf" \
    || { echo "  NEAT download failed; fetch it by hand"; rm -f "$neat"; }
fi
echo "done."
