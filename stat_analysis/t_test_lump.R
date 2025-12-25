df_glove <- df 

# paired t-test for each (location, size) combo:
tt_loc_size <- df_glove %>%
  group_by(True_S, True_L) %>%        # each location–size interaction
  t_test(
    Accuracy ~ Glove_Mode,            # with vs without glove
    paired = TRUE,                     # same Participant in both conditions
    alternative = "two.sided"
  ) %>%
  adjust_pvalue(method = "bonferroni") %>%  # optional: multiple-comparison correction
  add_significance()

tt_loc_size
