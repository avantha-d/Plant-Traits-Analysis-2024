# # from sklearn.ensemble import IsolationForest

# # def detect_outliers(features):
# #     # Use Isolation Forest for outlier detection
# #     clf = IsolationForest(contamination=0.1, random_state=42)
# #     outliers = clf.fit_predict(features)
# #     return outliers

# Extract predictions on validation dataset
valid_pred = model.predict(valid_ds)

# Calculate residuals
residuals = np.abs(valid_pred["head"] - valid_labels)

# Compute MEPA score for each sample
MEPA_scores = np.max(residuals, axis=1)

# Determine threshold for MEPA scores
threshold = np.percentile(MEPA_scores, 95)  # Example: 95th percentile

# Identify outliers
outliers_indices = np.where(MEPA_scores > threshold)[0]
outliers = valid_df.iloc[outliers_indices]

# Print outliers
print("Detected outliers:")
print(outliers)

