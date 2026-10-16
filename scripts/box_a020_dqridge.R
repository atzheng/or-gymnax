# Boxplot of LSTD-DQ(λ) final estimates over the ridge grid, for λ in {0, 0.8}, next to Naive
# (xp_gym vast/dqridge-a020-b*.yaml; dqridge_pair0.2:{B}.csv). Evaluation trajectories only (the
# first 10 (trial, env_id) are for tuning). For each λ, the ridge with the least mean relative MSE
# (MSE / ATE^2, averaged over the available settings) on the tuning trajectories is marked with ★.
# Ridges >= 0.1 shrink DQ back to Naive and are left out.
#
# usage: Rscript scripts/box_a020_dqridge.R [dir] [out_prefix]
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[1] else "data/a020/paper"
prefix <- if (length(args) >= 2) args[2] else "a020_dqridge"
n_tune <- 10

settings <- tibble(
  B   = c("0.26", "0.15", "0.025"),
  ate = c(0.587, -1.225, -6.094),
  pct = c(0.49, -1.01, -5.04)
) |>
  filter(file.exists(file.path(dir, sprintf("dqridge_pair0.2:%s.csv", B)))) |>
  mutate(setting = factor(sprintf("ATE %+.2f%%", pct), levels = sprintf("ATE %+.2f%%", pct)))

lams <- c("0.0", "0.8")
ridges <- c("1e-06", "0.0001", "0.001", "0.01")
grid <- expand_grid(lam = lams, ridge = ridges) |>
  mutate(column = sprintf("lstd_tr%s_r%s", lam, ridge),
         label = sprintf("λ=%g, ridge %s", as.numeric(lam), format(as.numeric(ridge), scientific = TRUE)))
cols <- c("naive", grid$column)

d <- bind_rows(lapply(seq_len(nrow(settings)), function(i) {
  read_csv(file.path(dir, sprintf("dqridge_pair0.2:%s.csv", settings$B[i])), show_col_types = FALSE,
           col_select = all_of(c("env_id", "trial", "steps", cols))) |>
    filter(steps == max(steps)) |>
    arrange(trial, env_id) |>
    mutate(traj = row_number()) |>
    pivot_longer(all_of(cols), names_to = "column", values_to = "estimate") |>
    mutate(setting = settings$setting[i], ate = settings$ate[i])
})) |>
  left_join(select(grid, column, lam, label), by = "column")

# tuned ridge per λ, one value across settings
picks <- d |>
  filter(traj <= n_tune, column != "naive") |>
  group_by(lam, column, setting) |>
  summarise(rel_mse = mean((estimate - ate)^2) / first(ate)^2, .groups = "drop") |>
  group_by(lam, column) |>
  summarise(score = mean(rel_mse), .groups = "drop") |>
  group_by(lam) |>
  slice_min(score, n = 1, with_ties = FALSE) |>
  ungroup()
print(as.data.frame(picks))

lv <- c("Naive", ifelse(grid$column %in% picks$column, paste0(grid$label, " ★"), grid$label))
d <- d |>
  filter(traj > n_tune) |>
  mutate(estimator = case_when(column == "naive" ~ "Naive",
                               column %in% picks$column ~ paste0(label, " ★"),
                               TRUE ~ label),
         estimator = factor(estimator, levels = lv),
         family = ifelse(column == "naive", "Naive", sprintf("λ = %g", as.numeric(lam))))

truth <- distinct(settings, setting, ate)
p <- ggplot(d, aes(estimator, estimate, color = family, fill = family)) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey50") +
  geom_hline(data = truth, aes(yintercept = ate), color = "red", linewidth = 0.6) +
  geom_boxplot(alpha = 0.2, outlier.size = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 2.5) +
  facet_wrap(~setting, nrow = 1, scales = "free_y") +
  labs(x = NULL, y = "estimate at t = 5e5", color = NULL, fill = NULL) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom")
ggsave(paste0("box_", prefix, ".png"), p, width = max(6, 4.2 * nrow(settings)), height = 5.5,
       dpi = 150, bg = "white")

summ <- d |>
  group_by(setting, ate, estimator) |>
  summarise(mean = mean(estimate), sd = sd(estimate), rmse = sqrt(mean((estimate - ate)^2)),
            bias_pct = 100 * (mean(estimate) - first(ate)) / abs(first(ate)),
            right_sign = mean(sign(estimate) == sign(ate)), .groups = "drop")
write_csv(summ, paste0("summary_", prefix, ".csv"))
