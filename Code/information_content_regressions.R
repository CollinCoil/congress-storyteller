# Mixed-effects regressions of information content (Tables 4 and 5).
#
# Input:   Data/witness_testimony_content.xlsx, Data/hearing_testimony_content.xlsx
# Output:  Model summaries, ICC, and Nakagawa R2 (printed); diagnostic plots
# Run from the repository root:  Rscript Code/information_content_regressions.R

# Load necessary libraries
library(readxl)
library(lme4)        # for mixed effects models
library(lmerTest)    # adds p-values to lmer
library(performance) # for ICC, diagnostics

# Open the Excel file
# Run from the repository root so the relative paths resolve
data <- read_excel("Data/witness_testimony_content.xlsx")

# Create the "Subcommittee_Level" column
data$Subcommittee_Level <- ifelse(is.na(data$Subcommittee) | data$Subcommittee == "", 0, 1)
data$Subcommittee_Level <- as.factor(data$Subcommittee_Level)

# Convert other relevant columns to factors if they are not already
data$Gender <- as.factor(data$Gender)
data$Hearing_Type <- as.factor(data$Hearing_Type)
data$Occupation <- as.factor(data$Occupation)
data$Narrative <- as.factor(data$Narrative)
data$Hearing_ID <- as.factor(paste(data$Date, data$Hearing, sep = "_"))
data$Committee_ID <- as.factor(paste(data$Branch, data$Committee, sep = "_"))


####################################################################
# Witness-Level Regression (Table 4)
####################################################################

# Random intercepts for committee and for hearing nested in committee
m1 <- lmer(log(Info_Content) ~ Gender + Hearing_Type + Occupation + Narrative +
                     Subcommittee_Level + (1 | Committee_ID / Hearing_ID),
                   data = data, REML = FALSE)

# Diagnostics
best_model <- m1

plot(fitted(best_model), residuals(best_model),
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Residuals vs. Fitted")
abline(h = 0, col = "red", lty = 2)

cat("Plotting QQ-plot of residuals...\n")
qqnorm(residuals(best_model))
qqline(residuals(best_model), col = "red")

cat("\nChecking Normality of Random Intercepts for each group...\n")
ranef_hearing <- ranef(best_model)$'Hearing_ID:Committee_ID'
qqnorm(ranef_hearing[[1]], main = "QQ-plot of Hearing Random Effects")
qqline(ranef_hearing[[1]], col = "red")

best_model_reml <- update(best_model, REML = TRUE)
cat("\n--- Final Model Summary (REML) ---\n")
summary(best_model_reml)

icc(best_model_reml)
VarCorr(best_model_reml)
r2_nakagawa(best_model_reml)


####################################################################
# Hearing-Level Regression (Table 5)
####################################################################

# Open the Excel file
hearing_data <- read_excel("Data/hearing_testimony_content.xlsx")

hearing_data$Percent_Women_Sq <- hearing_data$Percent_Women**2

# Create the "Subcommittee_Level" column
hearing_data$Subcommittee_Level <- ifelse(is.na(hearing_data$Subcommittee) | hearing_data$Subcommittee == "", 0, 1)
hearing_data$Subcommittee_Level <- as.factor(hearing_data$Subcommittee_Level)

hearing_data$Committee_ID <- as.factor(paste(hearing_data$Branch, hearing_data$Committee, sep = "_"))


# Perform the linear regression at the hearing level
model_hearing_re <- lmer(log(Info_Content) ~ Percent_Women + Percent_Women_Sq +
                           I(-1*Herfindahl_Index) + Percent_Narratives + I(Percent_Narratives**2) +
                           Hearing_Type + Num_Witnesses + Subcommittee_Level +
                           (1 | Committee_ID),
                         data = hearing_data, REML = TRUE)


plot(fitted(model_hearing_re), residuals(model_hearing_re),
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Residuals vs. Fitted")
abline(h = 0, col = "red", lty = 2)

cat("Plotting QQ-plot of residuals...\n")
qqnorm(residuals(model_hearing_re))
qqline(residuals(model_hearing_re), col = "red")

cat("\nChecking Normality of Random Intercepts for each group...\n")
ranef_hearing <- ranef(model_hearing_re)$'Committee_ID'
qqnorm(ranef_hearing[[1]], main = "QQ-plot of Hearing Random Effects")
qqline(ranef_hearing[[1]], col = "red")

cat("\n--- Final Model Summary (REML) ---\n")
summary(model_hearing_re)

icc(model_hearing_re)
VarCorr(model_hearing_re)
r2_nakagawa(model_hearing_re)
