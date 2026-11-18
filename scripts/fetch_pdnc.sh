#!/usr/bin/env bash
# Download one book from PDNC (Project Dialogism Novel Corpus, CC-BY 4.0,
# https://github.com/Priya22/project-dialogism-novel-corpus) into data/pdnc/<Book>/.
# Evaluation only: never feed these files into the pipeline (see docs/DESIGN.md).
# usage: scripts/fetch_pdnc.sh [PrideAndPrejudice]
set -euo pipefail
cd "$(dirname "$0")/.."
book="${1:-PrideAndPrejudice}"
base="https://raw.githubusercontent.com/Priya22/project-dialogism-novel-corpus/master/data/$book"
mkdir -p "data/pdnc/$book"
for f in character_info.csv quotation_info.csv novel_text.txt; do
  curl -fsSL "$base/$f" -o "data/pdnc/$book/$f"
done
echo "PDNC (CC-BY 4.0, Vishnubhotla et al. 2022): $base" > "data/pdnc/$book/SOURCE.txt"
ls -l "data/pdnc/$book"
