# Prepend a short densely-estimated run (first 10k steps, estimates every 1k) to the
# paper-protocol CSVs of one setting, for plotting the start of the RMSE-vs-t curves.
# NB: the short run is a fresh simulation (the step keys depend on the chunking), so its
# trajectories differ from the long run's; only the t < 10k rows come from it.
#
# usage: Rscript scripts/stitch_a020_early.R <early_dir> <B> [paper_dir] [out_dir]
#   early_dir: fork_/ope_/tsri_ CSVs of the short run (n_steps=1e4, estimate_every_n_steps=1e3)
#   paper_dir (default data/a020/paper) -> out_dir (default data/a020/paper_stitched): a copy of
#   paper_dir with B's three CSVs stitched. Every output row gets an explicit `t` (steps simulated).
suppressPackageStartupMessages({ library(dplyr); library(readr) })
args <- commandArgs(trailingOnly = TRUE)
early <- args[1]; B <- args[2]
paper <- if (length(args) >= 3) args[3] else "data/a020/paper"
out <- if (length(args) >= 4) args[4] else "data/a020/paper_stitched"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
for (f in list.files(paper, pattern = "\\.csv$")) file.copy(file.path(paper, f), file.path(out, f), overwrite = TRUE)
for (f in sprintf(c("fork_pair0.2:%s_seed42.csv", "ope_pair0.2:%s.csv", "tsri_pair0.2:%s_mb.csv"), B)) {
  long <- read_csv(file.path(paper, f), show_col_types = FALSE) |> mutate(t = steps + 10000)
  short <- read_csv(file.path(early, f), show_col_types = FALSE) |> mutate(t = steps + 1000) |>
    filter(t < min(long$t)) |> select(any_of(names(long)))
  write_csv(bind_rows(short, long) |> arrange(trial, env_id, t), file.path(out, f))
  cat(sprintf("%s: %d short rows (t = %s) + %d long rows\n", f, nrow(short),
              paste(sort(unique(short$t)), collapse = ","), nrow(long)))
}
