import pandas as pd
import numpy as np
from datetime import datetime

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------------------
# 1. Load the Datasets
# ---------------------------
wildfire_file = "historical_wildfiredata.csv"
env_file = "historical_environmental_data.csv"

wildfire_df = pd.read_csv(wildfire_file)
env_df = pd.read_csv(env_file)

# ---------------------------
# 2. Preprocess the Wildfire Data
# ---------------------------
wildfire_df['fire_start_time'] = pd.to_datetime(wildfire_df['fire_start_time'])
wildfire_df['rounded_fire_time'] = wildfire_df['fire_start_time'].dt.floor('H')

# Map severity from text to numeric: low -> 1, medium -> 2, high -> 3
severity_map = {'low': 1, 'medium': 2, 'high': 3}
wildfire_df['severity_numeric'] = wildfire_df['severity'].str.lower().map(severity_map)

# In case more than one fire occurred in the same hour, take the maximum severity (i.e., worst case)
agg_wildfire = wildfire_df.groupby('rounded_fire_time', as_index=False)['severity_numeric'].max()

# ---------------------------
# 3. Preprocess the Environmental Data
# ---------------------------
env_df['timestamp'] = pd.to_datetime(env_df['timestamp'])
# Environmental data is recorded hourly so no further rounding is needed.

# ---------------------------
# 4. Merge the Two Datasets
# ---------------------------
merged_df = pd.merge(env_df, agg_wildfire, how='left', left_on='timestamp', right_on='rounded_fire_time')

# Fill missing severity with 0 (i.e., no fire occurred in that hour)
merged_df['severity_numeric'] = merged_df['severity_numeric'].fillna(0).astype(int)

# Save the merged DataFrame to a CSV file (optional)
merged_df.to_csv("merged_data.csv", index=False)

# ---------------------------
# Transform to Binary Target
# ---------------------------
# For binary classification, set fire_occurred = 1 if severity_numeric > 0, else 0.
merged_df['fire_occurred'] = merged_df['severity_numeric'].apply(lambda x: 1 if x > 0 else 0)

# ---------------------------
# 5. Prepare Features and Target for Modeling
# ---------------------------
features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'vegetation_index', 'human_activity_index']
target = 'fire_occurred'

X = merged_df[features]
y = merged_df[target]

# Compute counts for binary classes
count_0 = (merged_df["fire_occurred"] == 0).sum()
count_1 = (merged_df["fire_occurred"] == 1).sum()
count_total = count_0 + count_1

custom_class_weights = {
    0: count_total / count_0 if count_0 > 0 else 1.0,
    1: count_total / count_1 if count_1 > 0 else 1.0,
}

print("Custom Class Weights:", custom_class_weights)

# ---------------------------
# 6. Train/Test Split and Model Training
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes in the training set
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=custom_class_weights)
model.fit(X_train_res, y_train_res)

# ---------------------------
# 7. Model Evaluation
# ---------------------------
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
