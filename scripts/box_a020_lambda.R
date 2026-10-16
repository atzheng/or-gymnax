# Boxplot of LSTD-DQ(λ) final estimates for each trace parameter λ, next to Naive, at A = 0.2.
# Same evaluation trajectories as scripts/plots_a020_paper.R (the 40 after the first 10
# (trial, env_id), which are reserved for tuning); no tuning here, every λ is shown.
#
# usage: Rscript scripts/box_a020_lambda.R [dir] [B,B,...] [out_prefix]
# dir defaults to data/a020/paper (fork_pair0.2:{B}_seed42.csv); fetch it with scripts/fetch_a020_data.sh
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[1] else "data/a020/paper"
Bs <- if (length(args) >= 2) strsplit(args[2], ",")[[1]] else c("0.26", "0.15", "0.025")
prefix <- if (length(args) >= 3) args[3] else "a020_lambda"
n_tune <- 10

settings <- tibble(
  B   = c("0.26", "0.175", "0.15", "0.1", "0.025"),
  ate = c(0.587, -0.537, -1.225, -2.799, -6.094),
  pct = c(0.49, -0.44, -1.01, -2.31, -5.04)
) |>
  filter(B %in% Bs) |>
  mutate(setting = factor(sprintf("ATE %+.2f%%", pct), levels = sprintf("ATE %+.2f%%", pct)))

lams <- c("0.0", "0.5", "0.8", "0.9", "0.95", "0.97")
cols <- c("naive", paste0("lstd_tr", lams))
labs_ <- setNames(c("Naive", paste0("λ = ", as.numeric(lams))), cols)

d <- bind_rows(lapply(seq_len(nrow(settings)), function(i) {
  read_csv(file.path(dir, sprintf("fork_pair0.2:%s_seed42.csv", settings$B[i])), show_col_types = FALSE,
           col_select = all_of(c("env_id", "trial", "steps", cols))) |>
    filter(steps == max(steps)) |>
    arrange(trial, env_id) |>
    mutate(traj = row_number()) |>
    filter(traj > n_tune) |>
    pivot_longer(all_of(cols), names_to = "column", values_to = "estimate") |>
    mutate(setting = settings$setting[i], ate = settings$ate[i])
})) |>
  mutate(estimator = factor(labs_[column], levels = labs_))

truth <- distinct(settings, setting, ate)
p <- ggplot(d, aes(estimator, estimate, color = estimator, fill = estimator)) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey50") +
  geom_hline(data = truth, aes(yintercept = ate), color = "red", linewidth = 0.6) +
  geom_boxplot(alpha = 0.2, outlier.size = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 2.5) +
  facet_wrap(~setting, nrow = 1, scales = "free_y") +
  labs(x = NULL, y = "estimate at t = 5e5") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")
ggsave(paste0("box_", prefix, ".png"), p, width = max(5, 3.4 * nrow(settings)), height = 5,
       dpi = 150, bg = "white")

summ <- d |>
  group_by(setting, ate, estimator) |>
  summarise(mean = mean(estimate), sd = sd(estimate), rmse = sqrt(mean((estimate - ate)^2)),
            bias_pct = 100 * (mean(estimate) - first(ate)) / abs(first(ate)),
            right_sign = mean(sign(estimate) == sign(ate)), .groups = "drop")
write_csv(summ, paste0("summary_", prefix, ".csv"))
print(as.data.frame(mutate(summ, across(where(is.numeric), \(x) round(x, 2)))), right = FALSE)
