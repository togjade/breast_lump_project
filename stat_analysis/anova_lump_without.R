### ANOVA_TEST ######
res.aov <- anova_test(
  data   = df_without,
  dv     = Accuracy,
  wid    = Participant,
  within = c(True_L, True_S),
  effect.size = "ges",
  type=3
)
tab <- get_anova_table(res.aov)
#tab <- get_anova_table(res.aov, correction = "GG")
tab
#### POST HOC #######
## LOCATION ##
pwc_L <- df_without %>%
  pairwise_t_test(
    Accuracy ~ True_L,
    paired = TRUE,
    p.adjust.method = "bonferroni"
  )
pwc_L
## SIZE ##
pwc_S <- df_without %>%
  pairwise_t_test(
    Accuracy ~ True_S,
    paired = TRUE,
    p.adjust.method = "bonferroni"
  )
pwc_S
#### POST HOC INTERACTION#######
pwc_L_by_S <- df_without %>%
  group_by(True_S) %>%                          # for each size separately
  pairwise_t_test(
    Accuracy ~ True_L,
    paired = TRUE,
    p.adjust.method = "bonferroni"
  ) %>%
  add_xy_position(x = "True_L", dodge = 0.8)

# keep only significant rows and avoid 'position' column name clash
pwc_L_by_S_sig <- pwc_L_by_S %>%
  filter(!is.na(p.adj), p.adj.signif != "ns") %>%  # only significant
  select(-matches("^position$"))

pwc_L_by_S_sig
#####
pwc_S_by_L <- df_without %>%
  group_by(True_L) %>%                          # for each size separately
  pairwise_t_test(
    Accuracy ~ True_S,
    paired = TRUE,
    p.adjust.method = "bonferroni"
  ) %>%
  add_xy_position(x = "True_L", dodge = 0.8)

# keep only significant rows and avoid 'position' column name clash
pwc_S_by_L_sig <- pwc_S_by_L %>%
  filter(!is.na(p.adj), p.adj.signif != "ns") %>%  # only significant
  select(-matches("^position$"))

pwc_S_by_L_sig
#####

