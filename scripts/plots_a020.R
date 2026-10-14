# Boxplot of final estimates and RMSE-vs-t (with bootstrap CIs) at A = 0.2, for
# Naive, TSRI-1/2, OPE-LSTD/GTD/TD and LSTD-DQ(lambda). Held-out seed 42, 50 envs.
#
# usage: Rscript scripts/plots_a020.R <dir> [out_prefix]
# <dir> holds the same CSVs as scripts/box_a020_below.py:
#   fork_pair0.2:{B}_seed42.csv, ope_pair0.2:{B}.csv, tsri_pair0.2:{B}_mb1.csv
# Estimates are saved every 10k steps; the row with steps = s is the estimate after
# s + 10k steps, so t runs 1e4 .. 5e5.
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr); library(purrr)
})

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[1] else "."
prefix <- if (length(args) >= 2) args[2] else "a020"
chunk <- 10000

settings <- tibble(
  B = c("0.26", "0.175", "0.15", "0.1"),
  ate = c(0.587, -0.537, -1.225, -2.799),
  pct = c(0.49, -0.44, -1.01, -2.31)
) |>
  mutate(setting = sprintf("B = %s: ATE %+.2f (%+.2f%%)", B, ate, pct),
         setting = factor(setting, levels = setting))

# column -> display name, and which file it lives in
ests <- tribble(
  ~file,  ~column,               ~estimator,
  "fork", "naive",               "Naive",
  "tsri", "tsri_tsri1_b0.3679",  "TSRI-1",
  "tsri", "tsri_tsri2_b0.3679",  "TSRI-2",
  "ope",  "ope_lstd_cf_r1e-08",  "OPE-LSTD",
  "ope",  "ope_gq_cf_sel",       "OPE-GTD",
  "ope",  "ope_td_cf_sel",       "OPE-TD",
  "fork", "lstd_tr0.5",          "LSTD-DQ(λ)"
)
files <- c(fork = "fork_pair0.2:%s_seed42.csv", ope = "ope_pair0.2:%s.csv",
           tsri = "tsri_pair0.2:%s_mb1.csv")

d <- pmap_dfr(settings, function(B, ate, pct, setting) {
  map_dfr(names(files), function(f) {
    cols <- ests$column[ests$file == f]
    read_csv(file.path(dir, sprintf(files[[f]], B)), show_col_types = FALSE,
             col_select = all_of(c("env_id", "trial", "steps", cols))) |>
      pivot_longer(all_of(cols), names_to = "column", values_to = "estimate")
  }) |>
    mutate(setting = setting, ate = ate)
}) |>
  left_join(select(ests, column, estimator), by = "column") |>
  mutate(estimator = factor(estimator, levels = ests$estimator),
         t = steps + chunk)

# ---- boxplot of final estimates ------------------------------------------
final <- filter(d, t == max(t))
truth <- distinct(final, setting, ate)
p_box <- ggplot(final, aes(estimator, estimate, color = estimator, fill = estimator)) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey50") +
  geom_hline(data = truth, aes(yintercept = ate), color = "red", linewidth = 0.6) +
  geom_boxplot(alpha = 0.2, outlier.size = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 2.5) +
  facet_wrap(~setting, nrow = 1, scales = "free_y") +
  labs(x = NULL, y = "estimate at t = 5e5",
       title = "ATE estimates at A = 0.2 (red line: true ATE; diamond: mean)",
       subtitle = "Held-out seed 42, 50 envs") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")
ggsave(paste0("box_", prefix, ".png"), p_box, width = 15, height = 5.5, dpi = 150, bg = "white")

# ---- RMSE vs t with percentile-bootstrap CIs over envs ----------------------
set.seed(1)
nboot <- 2000
boot_rmse <- function(err) {
  sq <- err^2
  n <- length(sq)
  idx <- matrix(sample.int(n, n * nboot, replace = TRUE), nrow = n)
  b <- sqrt(colMeans(matrix(sq[idx], nrow = n)))
  tibble(rmse = sqrt(mean(sq)), lo = quantile(b, 0.025), hi = quantile(b, 0.975), n = n)
}
rmse <- d |>
  filter(!is.na(estimate)) |>
  group_by(setting, ate, estimator, t) |>
  summarise(boot_rmse(estimate - ate), .groups = "drop")
write_csv(select(rmse, -ate), paste0("rmse_vs_t_", prefix, ".csv"))

tbreaks <- c(1e4, 3e4, 1e5, 3e5, 5e5)
p_rmse <- ggplot(rmse, aes(t, rmse, color = estimator, fill = estimator)) +
  geom_hline(data = truth, aes(yintercept = abs(ate)), linetype = "dashed", color = "red") +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.7) +
  scale_x_log10(breaks = tbreaks, labels = c("1e4", "3e4", "1e5", "3e5", "5e5")) +
  scale_y_log10() +
  facet_wrap(~setting, nrow = 1, scales = "free_y") +
  labs(x = "steps t", y = "RMSE vs true ATE",
       title = "RMSE vs steps at A = 0.2 (dashed red: |ATE|)",
       subtitle = "Held-out seed 42, 50 envs; bands are 95% bootstrap CIs over envs",
       color = NULL, fill = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave(paste0("rmse_vs_t_", prefix, ".png"), p_rmse, width = 15, height = 5.5, dpi = 150, bg = "white")

# table at the requested t
rmse |>
  filter(t %in% tbreaks) |>
  mutate(cell = sprintf("%.2f [%.2f, %.2f]", rmse, lo, hi)) |>
  select(setting, estimator, t, cell) |>
  pivot_wider(names_from = t, values_from = cell) |>
  arrange(setting, estimator) |>
  as.data.frame() |>
  print(right = FALSE)
