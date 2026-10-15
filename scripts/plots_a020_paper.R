# Paper protocol at A = 0.2: hyperparameters chosen to minimize MSE (vs the true ATE, at the
# final step) on 10 tuning trajectories; plots and tables use the other 40 trajectories.
# Held-out seed 42, 50 trajectories = 25 envs x 2 trials; tuning = first 10 (trial, env_id).
#
# usage: Rscript scripts/plots_a020_paper.R [dir] [out_prefix]
# dir defaults to data/a020/paper; fetch it with scripts/fetch_a020_data.sh
# <dir>: fork_pair0.2:{B}_seed42.csv (Naive, LSTD-DQ), and from s3://research/dq/paper/<job>/
#        ope_pair0.2:{B}.csv and tsri_pair0.2:{B}_mb.csv
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr); library(purrr)
})

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[1] else "data/a020/paper"
prefix <- if (length(args) >= 2) args[2] else "a020_paper"
chunk <- 10000
n_tune <- 10

settings <- tibble(
  B   = c("0.26", "0.175", "0.15", "0.1", "0.025", "0.49"),
  ate = c(0.587, -0.537, -1.225, -2.799, -6.094, -6.108),
  pct = c(0.49, -0.44, -1.01, -2.31, -5.04, -5.05)
) |>
  mutate(setting = sprintf("B = %s: ATE %+.2f (%+.2f%%)", B, ate, pct),
         setting = factor(setting, levels = setting))

grid_td <- expand_grid(g = c("0.001", "0.01", "0.1", "1"), b = c("0.2", "0.5"))
# family -> file and candidate columns (the hyperparameter grid); one column = no tuning
fams <- list(
  "Naive"      = list(file = "fork", cols = "naive"),
  "TSRI-1"     = list(file = "tsri", cols = "tsri_tsri1_b0.1983"),
  "TSRI-2"     = list(file = "tsri", cols = "tsri_tsri2_b0.1983"),
  "OPE-LSTD"   = list(file = "ope",  cols = "ope_lstd_cf_r1e-08"),
  "OPE-GTD"    = list(file = "ope",  cols = paste0("ope_gqit_cf_a", c("0.001", "0.01", "0.1", "1"))),
  "OPE-TD"     = list(file = "ope",  cols = sprintf("ope_tdit_cf_a%s_b%s", grid_td$g, grid_td$b)),
  "LSTD-DQ(λ)" = list(file = "fork", cols = paste0("lstd_tr", c("0.0", "0.5", "0.8", "0.9", "0.95", "0.97")))
)
files <- c(fork = "fork_pair0.2:%s_seed42.csv", ope = "ope_pair0.2:%s.csv",
           tsri = "tsri_pair0.2:%s_mb.csv")

long <- pmap_dfr(settings, function(B, ate, pct, setting) {
  map_dfr(names(files), function(f) {
    cols <- unique(unlist(map(fams, \(x) if (x$file == f) x$cols)))
    d <- read_csv(file.path(dir, sprintf(files[[f]], B)), show_col_types = FALSE,
                  col_select = any_of(c("env_id", "trial", "steps", "t", cols)))
    # stitched files (scripts/stitch_a020_early.R) carry t; otherwise t = steps + chunk
    if (!"t" %in% names(d)) d$t <- d$steps + chunk
    d |>
      arrange(trial, env_id) |>
      mutate(traj = match(paste(trial, env_id), unique(paste(trial, env_id)))) |>
      pivot_longer(all_of(cols), names_to = "column", values_to = "estimate")
  }) |>
    mutate(setting = setting, ate = ate)
})
fam_of <- map_dfr(names(fams), \(n) tibble(estimator = n, column = fams[[n]]$cols))
long <- inner_join(long, fam_of, by = "column", relationship = "many-to-many") |>
  mutate(estimator = factor(estimator, levels = names(fams)))

# ---- tuning: per setting and estimator, the column with least MSE on the 10 tuning trajectories
tmax <- max(long$t)
picks <- long |>
  filter(t == tmax, traj <= n_tune) |>
  group_by(setting, estimator, column) |>
  summarise(tune_mse = mean((estimate - ate)^2), .groups = "drop_last") |>
  slice_min(tune_mse, n = 1, with_ties = FALSE) |>
  ungroup()
write_csv(picks, paste0("picks_", prefix, ".csv"))
cat("Hyperparameters picked on", n_tune, "tuning trajectories:\n")
print(as.data.frame(select(picks, setting, estimator, column)), right = FALSE)

d <- semi_join(long, picks, by = c("setting", "estimator", "column")) |>
  filter(traj > n_tune)
truth <- distinct(settings, setting, ate)

# ---- boxplot of final estimates (evaluation trajectories)
final <- filter(d, t == tmax)
p_box <- ggplot(final, aes(estimator, estimate, color = estimator, fill = estimator)) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey50") +
  geom_hline(data = truth, aes(yintercept = ate), color = "red", linewidth = 0.6) +
  geom_boxplot(alpha = 0.2, outlier.size = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 2.5) +
  facet_wrap(~setting, nrow = 2, scales = "free_y") +
  labs(x = NULL, y = "estimate at t = 5e5",
       title = "ATE estimates at A = 0.2 (red line: true ATE; diamond: mean)",
       subtitle = sprintf("Hyperparameters tuned on %d trajectories; %d evaluation trajectories shown (seed 42)",
                          n_tune, length(unique(final$traj)))) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")
ggsave(paste0("box_", prefix, ".png"), p_box, width = 13, height = 9, dpi = 150, bg = "white")

# ---- RMSE vs t with percentile-bootstrap CIs over evaluation trajectories
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

tbreaks <- c(1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 5e5)
p_rmse <- ggplot(rmse, aes(t, rmse, color = estimator, fill = estimator)) +
  geom_hline(data = truth, aes(yintercept = abs(ate)), linetype = "dashed", color = "red") +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.7) +
  scale_x_log10(breaks = tbreaks, labels = c("1e3", "3e3", "1e4", "3e4", "1e5", "3e5", "5e5")) +
  scale_y_log10() +
  facet_wrap(~setting, nrow = 2, scales = "free_y") +
  labs(x = "steps t", y = "RMSE vs true ATE",
       title = "RMSE vs steps at A = 0.2 (dashed red: |ATE|)",
       subtitle = sprintf("Tuned on %d trajectories; bands are 95%% bootstrap CIs over the other %d",
                          n_tune, max(rmse$n)),
       color = NULL, fill = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave(paste0("rmse_vs_t_", prefix, ".png"), p_rmse, width = 13, height = 9, dpi = 150, bg = "white")

# ---- summary table at the final step (evaluation trajectories)
summ <- final |>
  group_by(setting, ate, estimator, column) |>
  summarise(mean = mean(estimate), rmse = sqrt(mean((estimate - ate)^2)),
            bias_pct = 100 * (mean(estimate) - first(ate)) / abs(first(ate)),
            right_sign = mean(sign(estimate) == sign(ate)), .groups = "drop")
write_csv(summ, paste0("summary_", prefix, ".csv"))
print(as.data.frame(mutate(summ, across(where(is.numeric), \(x) round(x, 2)))), right = FALSE)
