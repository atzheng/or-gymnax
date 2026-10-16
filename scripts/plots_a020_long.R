# RMSE vs t up to 1e7 steps at A = 0.2 from the long run (xp_gym config long_expts,
# s3://research/dq/long/<job>/long_pair0.2:{B}.csv). Only B = 0.175 finished
# (wakeful-married-slug-of-competence); the other long runs were cancelled.
# Fixed settings (no tuning): LSTD-DQ lambda = 0.5, OPE-LSTD ridge 1e-8, and the online
# OPE estimators with constant step alpha = 0.01 (TD Rbar ratio 1, Diff-GQ1 helper ratio 1).
# All 50 trajectories (25 envs x 2 trials); estimates every 1e4 steps, so t = steps + 1e4.
#
# usage: Rscript scripts/plots_a020_long.R [dir] [B] [out_prefix]
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[1] else "data/a020/long"
B <- if (length(args) >= 2) args[2] else "0.175"
prefix <- if (length(args) >= 3) args[3] else paste0("a020_long_", B)
chunk <- 10000
ate <- c("0.26" = 0.587, "0.175" = -0.537, "0.15" = -1.225, "0.1" = -2.799)[[B]]

ests <- c("naive" = "Naive", "ope_lstd_cf_r1e-08" = "OPE-LSTD", "ope_gqit_cf_a0.01" = "OPE-GTD",
          "ope_tdit_cf_a0.01" = "OPE-TD", "lstd_tr0.5" = "LSTD-DQ(λ)")
# same colours as scripts/plots_a020_paper.R (default hue over its 7 estimators)
all_ests <- c("Naive", "TSRI-1", "TSRI-2", "OPE-LSTD", "OPE-GTD", "OPE-TD", "LSTD-DQ(λ)")
pal <- setNames(scales::hue_pal()(length(all_ests)), all_ests)

d <- read_csv(file.path(dir, sprintf("long_pair0.2:%s.csv", B)), show_col_types = FALSE,
              col_select = all_of(c("env_id", "trial", "steps", names(ests)))) |>
  pivot_longer(all_of(names(ests)), names_to = "column", values_to = "estimate") |>
  mutate(estimator = factor(ests[column], levels = intersect(all_ests, ests)),
         t = steps + chunk)

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
  group_by(estimator, t) |>
  summarise(boot_rmse(estimate - ate), .groups = "drop")
write_csv(rmse, paste0("rmse_vs_t_", prefix, ".csv"))

tbreaks <- c(1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7)
p <- ggplot(rmse, aes(t, rmse, color = estimator, fill = estimator)) +
  geom_hline(yintercept = abs(ate), linetype = "dashed", color = "red") +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.7) +
  scale_color_manual(values = pal) + scale_fill_manual(values = pal) +
  scale_x_log10(breaks = tbreaks, labels = c("1e4", "3e4", "1e5", "3e5", "1e6", "3e6", "1e7")) +
  scale_y_log10() +
  # OPE-GTD diverges in some trajectories late in the run; let it run off the top
  coord_cartesian(ylim = c(min(rmse$lo, na.rm = TRUE), 5 * max(rmse$hi[rmse$estimator == "Naive"]))) +
  labs(x = "steps t", y = "RMSE vs true ATE",
       title = sprintf("RMSE vs steps at A = 0.2, B = %s: ATE %+.2f (dashed red: |ATE|)", B, ate),
       subtitle = sprintf(paste0("Fixed settings: LSTD-DQ λ = 0.5; OPE-TD / OPE-GTD constant step α = 0.01\n",
                                 "OPE-GTD diverges after ~7e5 (off scale); bands are 95%% bootstrap CIs over %d trajectories"),
                          max(rmse$n)),
       color = NULL, fill = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave(paste0("rmse_vs_t_", prefix, ".png"), p, width = 8, height = 5.5, dpi = 150, bg = "white")

rmse |>
  filter(t %in% tbreaks) |>
  mutate(cell = sprintf("%.2f [%.2f, %.2f]", rmse, lo, hi)) |>
  select(estimator, t, cell) |>
  pivot_wider(names_from = t, values_from = cell) |>
  as.data.frame() |>
  print(right = FALSE)
