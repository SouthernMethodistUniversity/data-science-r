# ============================================================
# 0. SETUP AND PROJECT STRUCTURE
# ============================================================

install.packages(c("tidyverse", "palmerpenguins"))

library(tidyverse)
library(palmerpenguins)

set.seed(123)

# NOTE: Create these folders manually in your RStudio Project:
#   data/
#   outputs/

# Write the built-in dataset to CSV if it is not already present.

example_path <- "data/penguins.csv"

if (!file.exists(example_path)) {
  write_csv(penguins, example_path)
}

# ============================================================
# 1. DATA LOADING
# ============================================================

df_raw <- read_csv(example_path, show_col_types = FALSE)

str(df_raw)
head(df_raw, n=10)
summary(df_raw)

# Notice that this dataset needs cleaning:
#   - some body measurements have missing values
#   - categorical variables may need to be recoded as factors
#   - we can create a binary outcome for classification

# ============================================================
# 2. DATA CLEANING AND FEATURE CREATION
# ============================================================

# Keep the cleaning visible and intentional.
# The goal here is to show that data cleaning is a major part of the workflow.

df <- df_raw %>%
  select(species, island, bill_length_mm, bill_depth_mm, flipper_length_mm,
         body_mass_g, sex) %>%
  mutate(
    species = as.factor(species),
    island = as.factor(island),
    sex = as.factor(sex),
    large_body = ifelse(body_mass_g > median(body_mass_g, na.rm = TRUE), 1, 0)
  ) %>%
  filter(
    !is.na(bill_length_mm),
    !is.na(bill_depth_mm),
    !is.na(flipper_length_mm),
    !is.na(body_mass_g),
    !is.na(sex)
  )

str(df)
summary(df)

# Compare row counts before and after cleaning.
nrow(df_raw)
nrow(df)

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

ggplot(df, aes(x = body_mass_g)) +
  geom_histogram(bins = 12) +
  labs(title = "Distribution of Body Mass")

# GGally
ggplot(df, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point() +
  labs(title = "Body Mass vs Flipper Length")

ggplot(df, aes(x = log(bill_length_mm), y = body_mass_g)) +
  geom_point() +
  labs(title = "Body Mass vs Bill Length")

ggplot(df, aes(x = species, y = body_mass_g)) +
  geom_boxplot() +
  labs(title = "Body Mass by Species")

# ============================================================
# 4. LINEAR REGRESSION
# ============================================================

# Predict body mass using body measurements.

lm_model <- lm(body_mass_g ~ flipper_length_mm, data = df)
lm_model <- lm(body_mass_g ~ bill_length_mm + bill_depth_mm + flipper_length_mm, data = df)

cor(df$flipper_length_mm, df$body_mass_g)
cor(df$bill_length_mm, df$body_mass_g)

summary(lm_model)
coef(lm_model)

ggplot(df, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Linear Regression: Body Mass vs Flipper Length")

df$predicted = -6445.476 + (3.2293*df$bill_length_mm) + (17.836*df$bill_depth_mm) + (50.762 * df$flipper_length_mm)

ggplot(df, aes(x = flipper_length_mm, y = predicted)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Linear Regression: Predicted vs Flipper Length")

ggplot(df, aes(x = bill_length_mm, y = predicted)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Linear Regression: Predicted vs Bill Length")

ggplot(df, aes(x = bill_depth_mm, y = predicted)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Linear Regression: Predicted vs Bill Depth")

# ============================================================
# 5. EXTRAPOLATION
# ============================================================

# Extrapolation means predicting outside the observed range.
range(df$flipper_length_mm)

# Prediction inside observed range.
predict(lm_model, newdata = data.frame(
  bill_length_mm = 45,
  bill_depth_mm = 17,
  flipper_length_mm = 200
))

# Prediction outside observed range.
predict(lm_model, newdata = data.frame(
  bill_length_mm = 65,
  bill_depth_mm = 25,
  flipper_length_mm = 260
))

ggplot(df, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  geom_vline(xintercept = range(df$flipper_length_mm), linetype = "dashed") +
  labs(title = "Extrapolation Warning")

# ============================================================
# 7. TRAIN/TEST SPLIT AND MODEL EVALUATION
# ============================================================

set.seed(123)
train_index <- sample(seq_len(nrow(df)), size = floor(0.8 * nrow(df)))
train <- df[train_index, ]
test  <- df[-train_index, ]

lm_train <- lm(body_mass_g ~ flipper_length_mm, data = train)

# WRONG: evaluate on training data
train_preds_wrong <- predict(lm_train, newdata = train)
sqrt(mean((train$body_mass_g - train_preds_wrong)^2))

# CORRECT: evaluate on test data
test_preds <- predict(lm_train, newdata = test)
sqrt(mean((test$body_mass_g - test_preds)^2))

eval_df <- test
eval_df$predicted_body_mass_g <- test_preds

ggplot(eval_df, aes(x = body_mass_g, y = predicted_body_mass_g)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Test Set Predictions")

# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

# large_body = 1 means body mass is above the dataset median
set.seed(123)
train_index_cls <- sample(seq_len(nrow(df)), size = floor(0.8 * nrow(df)))
train_cls <- df[train_index_cls, ]
test_cls  <- df[-train_index_cls, ]

glm_model <- glm(large_body ~ bill_length_mm + bill_depth_mm + flipper_length_mm,
                 data = train_cls, family = "binomial")

summary(glm_model)

prob_preds <- predict(glm_model, newdata = test_cls, type = "response")
class_preds <- ifelse(prob_preds >= 0.5, 1, 0)

mean(class_preds == test_cls$large_body)

table(actual = test_cls$large_body, predicted = class_preds)

# ============================================================
# 9. REPRODUCIBILITY AND SCALING NOTES
# ============================================================

# Save outputs (ensure 'outputs/' exists)
write_csv(eval_df, "outputs/lm_test_predictions.csv")

# Example: run script
# Rscript script.R

# Key idea: the same workflow scales to larger data and HPC.
