library(readxl)
library(dplyr)
library(purrr)
library(emmeans)
library(afex)
library(readr)
library(rstatix)
library(reshape)
#library(tidyverse)
library(dplyr)
library(ggpubr)
library(plyr)
library(datarium)

folder <- "/media/tactile/Togzhan_disk2/files/project_files/lab_files/LAB_lump_project/data/moldir_blind_exp/"
  #"Downloads/"
csv_files <- list.files(folder, pattern = "\\.csv$", full.names = TRUE)
df_ <- read_csv("/media/tactile/Togzhan_disk2/files/project_files/lab_files/LAB_lump_project/data/moldir_blind_exp/df_glove_modes_type.csv")
df = df_[df_$acc_type=="acc_lp", ]
################## WITH AND WITHOUT CASE #########################################
df <- df %>%
  mutate(
    Participant = factor(Participant),
    Glove_Mode  = factor(Glove_Mode),  # "0" = no glove, "1" = glove
    True_L      = factor(True_L),
    True_S      = factor(True_S)
  )
################## WITHOUT CASE #########################################

df_without <- df[df$Glove_Mode==0,]
df_without <- df_without %>%
  mutate(
    Participant = factor(Participant),
    True_L      = factor(True_L, levels = c("Top", "Middle", "Bottom")),
    True_S      = factor(True_S, levels = c("Small", "Medium", "Large")),
    LS          = interaction(True_S, True_L, sep = "_")
  )
df_without$Accuracy <- as.numeric(df_without$Accuracy)

################## WITH CASE #########################################

df_with <- df[df$Glove_Mode==1,]
df_with <- df_with %>%
  mutate(
    Participant = factor(Participant),
    True_L      = factor(True_L, levels = c("Top", "Middle", "Bottom")),
    True_S      = factor(True_S, levels = c("Small", "Medium", "Large")),
    LS          = interaction(True_S, True_L, sep = "_")
  )
df_with$Accuracy <- as.numeric(df_with$Accuracy)

