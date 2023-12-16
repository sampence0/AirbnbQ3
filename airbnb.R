##################
# AIRBNB PROJECT #
##################

# Import Libraries
library(caret)
library(glmnet)


# load data
load("airbnb_project.rdata")

# EDA
print_column_names <- function(data, name) {
  cat("Column names in", name, ":\n")
  print(names(data))
  cat("\n")
}
print_column_names(listing_2016Q1, "listing_2016Q1")
print_column_names(listing_2016Q2, "listing_2016Q2")
print_column_names(property_info, "property_info")
print_column_names(reserve_2016Q3_train, "reserve_2016Q3_train")
print(PropertyID_test)

summarize_data <- function(data, name) {
  cat("Summary for", name, ":\n")
  print(summary(data))
  cat("Number of missing values in each column:\n")
  print(colSums(is.na(data)))
  cat("Data types:\n")
  print(sapply(data, class))
  cat("Number of unique values in key columns (if applicable):\n")
  if ("PropertyID" %in% names(data)) {
    print(length(unique(data$PropertyID)))
  }
  cat("\n")
}
summarize_data(listing_2016Q1, "listing_2016Q1")
summarize_data(listing_2016Q2, "listing_2016Q2")
summarize_data(property_info, "property_info")
summarize_data(reserve_2016Q3_train, "reserve_2016Q3_train")

# --- PREPROCESS --- #
# 'IsBooked' indicating whether or not property is booked on given date
listing_2016Q1$IsBooked <- as.integer(!is.na(listing_2016Q1$BookedDate))
listing_2016Q2$IsBooked <- as.integer(!is.na(listing_2016Q2$BookedDate))

# aggregate booking for each property ID by quarter
agg_bookings_Q1 <- aggregate(IsBooked ~ PropertyID, data = listing_2016Q1, FUN = sum)
agg_bookings_Q2 <- aggregate(IsBooked ~ PropertyID, data = listing_2016Q2, FUN = sum)
names(agg_bookings_Q1)[2] <- "NumReserveDays2016Q1"
names(agg_bookings_Q2)[2] <- "NumReserveDays2016Q2"
merged_data <- merge(property_info, agg_bookings_Q1, by = "PropertyID", all = TRUE)
merged_data <- merge(merged_data, agg_bookings_Q2, by = "PropertyID", all = TRUE)


# split to train/test
train_set <- merged_data[!merged_data$PropertyID %in% PropertyID_test, ]
test_set <- merged_data[merged_data$PropertyID %in% PropertyID_test, ]
train_set <- merge(train_set, reserve_2016Q3_train, by = "PropertyID", all.x = TRUE)
train_set$NumReserveDays2016Q3[is.na(train_set$NumReserveDays2016Q3)] <- 0

summarize_data(train_set, "train_set")
summarize_data(test_set, "test_set")


clean_data <- function(df) {
  df$CreatedDate <- NULL
  df$HostID <- NULL
  df <- cbind(df, model.matrix(~CancellationPolicy + ListingType - 1, data = df))
  df$CancellationPolicy <- NULL
  df$ListingType <- NULL
  df[df == "Yes"] <- 1
  df[df == "No"] <- 0
  df <- as.data.frame(lapply(df, function(x) if (is.logical(x)) as.integer(x) else x))
  numeric_cols <- sapply(df, is.numeric)
  df[numeric_cols] <- lapply(df[numeric_cols], function(x) {
    x[is.na(x)] <- median(x, na.rm = TRUE)
    return(x)
  })
  df <- df[numeric_cols | grepl("CancellationPolicy|ListingType", names(df))]
  
  # Add interaction terms
  df$Interaction_Rating <- with(df, OverallRating * AccuracyRating * CleanRating)
  df$Interaction_GuestsBedBath <- with(df, MaxGuests * Bedrooms * Bathrooms)
  df$Interaction_ResponseSuperhost <- with(df, ResponseRate * Superhost)
  df$Interaction_SecurityPolicy <- with(df, SecurityDeposit * CancellationPolicyStrict)
  df$Interaction_GuestsBedrooms <- with(df, MaxGuests * Bedrooms)
  df$Interaction_PriceResponseRate <- with(df, PublishedNightlyRate * ResponseRate)
  
  # Polynomial terms
  df$NumberofReviews2 <- df$NumberofReviews^2
  df$PublishedNightlyRate2 <- df$PublishedNightlyRate^2
  df$ResponseTimemin2 <- df$ResponseTimemin^2
  df$MinimumStay2 <- df$MinimumStay^2
  df$PriceSquared <- df$PublishedNightlyRate^2
  df$ReviewsSquared <- df$NumberofReviews^2
  df$ResponseRateCubed <- df$ResponseRate^3
  
  return(df)
}



# Apply cleaning function
train_set_processed <- clean_data(train_set)
test_set_processed <- clean_data(test_set)

# Remove Non-Refundable Cancellation Column (Not in test)
train_set_processed <- train_set_processed[, !names(train_set_processed) 
                                           %in% "CancellationPolicyNo.Refunds"]



# --- MODEL BUILDING --- #
model_sig <- lm(NumReserveDays2016Q3 ~ ., data = train_set_processed)
model_sig_summary <- summary(model_sig)
significant_features <- rownames(coef(model_sig_summary)[coef(model_sig_summary)[, "Pr(>|t|)"] < 0.05, ])
significant_features <- significant_features[significant_features != "(Intercept)"]
train_set_significant <- train_set_processed[, significant_features, drop = FALSE]
train_set_significant$NumReserveDays2016Q3 <- train_set_processed$NumReserveDays2016Q3

# Train with cross-validation
train_control_sig <- trainControl(method = "cv", number = 10)
model_sig_trained <- train(NumReserveDays2016Q3 ~ ., data = train_set_significant, 
                           method = "glm", trControl = train_control_sig)
print(model_sig_trained)


# Predict on test set
test_set_significant <- test_set_processed[, significant_features, drop = FALSE]
significant_features
names(test_set_processed)
test_predictions <- predict(model_sig_trained, test_set_significant)
pred <- test_predictions[match(PropertyID_test, test_set_processed$PropertyID)]

# Save 'pred' to an .rdata file
save(pred, file = "pred.rdata")


# Step Model
full_model <- lm(NumReserveDays2016Q3 ~ ., data = train_set_processed)
step_model <- step(full_model, direction = "backward", trace = 0) # 'trace = 0' to minimize output verbosity
summary(step_model)
train_control <- trainControl(method = "cv", number = 10) # 10-fold cross-validation
step_model_formula <- formula(step_model)
cv_model <- train(step_model_formula, 
                  data = train_set_processed, 
                  method = "lm", 
                  trControl = train_control)
print(cv_model)

# Predict on Test and save as pred_step
test_predictions_step <- predict(step_model, newdata = test_set_processed)
pred_step <- test_predictions_step[match(PropertyID_test, test_set_processed$PropertyID)]
save(pred_step, file = "pred_step.rdata")


pred_step

