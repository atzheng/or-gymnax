# Paper figures for the ride-hailing experiments at A = 0.2 (control savings threshold 0.2,
# treatment B in {0.26, 0.15, 0.025}).  Held-out seed 42, 50 trajectories = 25 envs x 2 trials;
# the first 10 (trial, env_id) are for tuning, the other 40 are shown.
#
# Hyperparameters: ONE value per estimator for all settings, minimizing the relative MSE
# (MSE / ATE^2, at the final step, averaged over the settings) on the tuning trajectories.
#   LSTD-DQ(λ): λ = 0.8 (fixed)       OPE-LSTD: ridge 1e-8 (no tuning)
#   OPE-TD: constant steps α/D, α x Rbar ratio β (tune_ope_pair0.2:{B}.csv)
#   OPE-GTD: Diff-GQ1, paper grid α (ope_pair0.2:{B}.csv) and α x helper ratio (tune file)
#   TSRI-1/2: market-balance design, k in {1, 2}, from the fixed TSR runs (tsri_fix_pair0.2:{B}_mb.csv)
#
# usage: Rscript scripts/plots_a020_final.R [dir] [out_prefix] [rmse_B]
# dir defaults to data/a020/paper; fetch it with scripts/fetch_a020_data.sh
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr); library(purrr)
})

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[1] else "data/a020/paper"
prefix <- if (length(args) >= 2) args[2] else "a020_final"
rmse_B <- if (length(args) >= 3) strsplit(args[3], ",")[[1]] else "0.025"
chunk <- 10000
n_tune <- 10

settings <- tibble(
  B   = c("0.26", "0.15", "0.025"),
  ate = c(0.587, -1.225, -6.094),
  pct = c(0.49, -1.01, -5.04)
) |>
  mutate(setting = factor(sprintf("ATE %+.2f%%", pct), levels = sprintf("ATE %+.2f%%", pct)))

al_tune <- c("0.003", "0.01", "0.03", "0.1", "0.2", "0.3", "0.5")
tune_td <- with(expand_grid(a = al_tune, b = c("0.03", "0.1", "0.3")), sprintf("a%s_b%s", a, b))
tune_gq <- with(expand_grid(a = al_tune, n = c("0.1", "1", "10")), sprintf("a%s_n%s", a, n))
# estimator -> (file, candidate columns); a column must exist in that file
fams <- tribble(
  ~estimator,   ~file,  ~column,
  "Naive",      "fork", "naive",
  "TSRI-1",     "tsri", "tsri_tsri1_b0.1983",
  "TSRI-2",     "tsri", "tsri_tsri2_b0.1983",
  "OPE-LSTD",   "ope",  "ope_lstd_cf_r1e-08",
  "LSTD-DQ(λ)", "fork", "lstd_tr0.8"
) |>
  bind_rows(tibble(estimator = "OPE-GTD", file = "ope",
                   column = paste0("ope_gqit_cf_a", c("0.001", "0.01", "0.1", "1"))),
            tibble(estimator = "OPE-GTD", file = "tune",
                   column = paste0("ope_", c("gq", "gqit"), "_cf_", rep(tune_gq, each = 2))),
            tibble(estimator = "OPE-TD", file = "tune",
                   column = paste0("ope_", c("td", "tdit"), "_cf_", rep(tune_td, each = 2))))
est_levels <- c("Naive", "TSRI-1", "TSRI-2", "OPE-LSTD", "OPE-GTD", "OPE-TD", "LSTD-DQ(λ)")
files <- c(fork = "fork_pair0.2:%s_seed42.csv", ope = "ope_pair0.2:%s.csv",
           tsri = "tsri_fix_pair0.2:%s_mb.csv", tune = "tune_ope_pair0.2:%s.csv")

long <- pmap_dfr(settings, function(B, ate, pct, setting) {
  map_dfr(names(files), function(f) {
    cols <- unique(fams$column[fams$file == f])
    read_csv(file.path(dir, sprintf(files[[f]], B)), show_col_types = FALSE,
             col_select = any_of(c("env_id", "trial", "steps", cols))) |>
      arrange(trial, env_id) |>
      mutate(traj = match(paste(trial, env_id), unique(paste(trial, env_id))), t = steps + chunk) |>
      pivot_longer(any_of(cols), names_to = "column", values_to = "estimate") |>
      mutate(file = f)
  }) |>
    mutate(setting = setting, ate = ate)
}) |>
  inner_join(fams, by = c("file", "column")) |>
  mutate(estimator = factor(estimator, levels = est_levels))

# ---- tuning: one column per estimator across settings (mean relative MSE on tuning trajectories)
tmax <- max(long$t)
scores <- long |>
  filter(t == tmax, traj <= n_tune) |>
  group_by(estimator, column, setting) |>
  summarise(rel_mse = mean((estimate - ate)^2) / first(ate)^2, .groups = "drop") |>
  group_by(estimator, column) |>
  summarise(n_set = n(), score = mean(rel_mse), .groups = "drop") |>
  filter(n_set == nrow(settings), is.finite(score))
picks <- scores |> group_by(estimator) |> slice_min(score, n = 1, with_ties = FALSE) |> ungroup()
write_csv(picks, paste0("picks_", prefix, ".csv"))
cat("Picked (one per estimator, all settings):\n"); print(as.data.frame(picks), right = FALSE)

d <- semi_join(long, picks, by = c("estimator", "column")) |> filter(traj > n_tune)
truth <- distinct(settings, setting, ate)

# ---- boxplot of final estimates
final <- filter(d, t == tmax)
p_box <- ggplot(final, aes(estimator, estimate, color = estimator, fill = estimator)) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey50") +
  geom_hline(data = truth, aes(yintercept = ate), color = "red", linewidth = 0.6) +
  geom_boxplot(alpha = 0.2, outlier.size = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 2.5) +
  facet_wrap(~setting, nrow = 1, scales = "free_y") +
  labs(x = NULL, y = "estimate at t = 5e5") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")
for (ext in c("png", "pdf"))
  ggsave(paste0("box_", prefix, ".", ext), p_box, width = 3.4 * nrow(settings), height = 5,
         dpi = 150, bg = "white", device = if (ext == "pdf") cairo_pdf else NULL)

# ---- RMSE (% of |ATE|) vs t with percentile-bootstrap CIs over evaluation trajectories
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
  group_by(setting, ate, estimator, column, t) |>
  summarise(boot_rmse(estimate - ate), .groups = "drop")
write_csv(select(rmse, -ate), paste0("rmse_vs_t_", prefix, ".csv"))

rmse_plot <- rmse |>
  filter(setting %in% settings$setting[settings$B %in% rmse_B]) |>
  mutate(across(c(rmse, lo, hi), \(v) 100 * v / abs(ate)))
tbreaks <- c(1e4, 3e4, 1e5, 3e5, 5e5)
p_rmse <- ggplot(rmse_plot, aes(t, rmse, color = estimator, fill = estimator)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.7) +
  scale_x_log10(breaks = tbreaks, labels = c("1e4", "3e4", "1e5", "3e5", "5e5")) +
  scale_y_log10(labels = \(x) paste0(format(x, big.mark = ",", scientific = FALSE, trim = TRUE), "%")) +
  facet_wrap(~setting, nrow = 1, scales = "free_y") +
  labs(x = "steps t", y = "RMSE (% of |ATE|)", color = NULL, fill = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
n_fac <- length(unique(rmse_plot$setting))
for (ext in c("png", "pdf"))
  ggsave(paste0("rmse_vs_t_", prefix, ".", ext), p_rmse, width = max(7, 3.4 * n_fac), height = 5.5,
         dpi = 150, bg = "white", device = if (ext == "pdf") cairo_pdf else NULL)

# ---- summary at the final step
summ <- final |>
  group_by(setting, ate, estimator, column) |>
  summarise(mean = mean(estimate), sd = sd(estimate), rmse = sqrt(mean((estimate - ate)^2)),
            bias_pct = 100 * (mean(estimate) - first(ate)) / abs(first(ate)),
            right_sign = mean(sign(estimate) == sign(ate)), .groups = "drop")
write_csv(summ, paste0("summary_", prefix, ".csv"))
print(as.data.frame(mutate(summ, across(where(is.numeric), \(x) round(x, 2)))), right = FALSE)
