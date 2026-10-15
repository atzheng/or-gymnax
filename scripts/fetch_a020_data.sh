#!/usr/bin/env bash
# Download the input CSVs for the A = 0.2 plots into data/a020/:
#   data/a020/untuned/  scripts/plots_a020.R (and box_a020_below.py, rmse_vs_t_a020.py)
#   data/a020/paper/    scripts/plots_a020_paper.R
#   data/a020/paper_stitched/  paper/ with B=0.025 extended to t=1e3..9e3 (scripts/stitch_a020_early.R)
#   data/a020/early_0.025/     the short run behind it (n_steps=1e4, estimates every 1e3)
# Needs AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY for the R2 bucket `research`.
set -euo pipefail
export AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL:-https://bc0a4d3391012c4fb52a9e8e78724da1.r2.cloudflarestorage.com}
root=$(cd "$(dirname "$0")/.." && pwd)
for d in untuned paper paper_stitched early_0.025; do
  aws s3 cp --recursive --only-show-errors "s3://research/dq/a020_inputs/$d/" "$root/data/a020/$d/"
done
ls "$root"/data/a020/*
