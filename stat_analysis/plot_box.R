### BOX SEPARATELKY
bxp_L <- ggboxplot(
  df_without,
  x        = "True_L",
  y        = "Accuracy",
  color    = "True_S",
  add      = "point",
)

bxp_L +
  stat_pvalue_manual(
    pwc_S_by_L_sig,
    label      = "p.adj.signif",
    tip.length = 0.01
  ) +
  labs(
    x        = "Location (True_L)",
    color    = "Size (True_S)",
    subtitle = get_test_label(res.aov, detailed = TRUE),
    caption  = get_pwc_label(pwc_S_by_L_sig)
  )
###
