import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import folium

# ---------------------------
# 1. Load the Datasets with Date Parsing
# ---------------------------
historical_env_data = pd.read_csv('historical_environmental_data.csv', parse_dates=['timestamp'])
historical_wildfire_data = pd.read_csv('historical_wildfiredata.csv', parse_dates=['timestamp', 'fire_start_time'])

# ---------------------------
# 2. Merge the Datasets and Create Binary Target
# ---------------------------
# Merge on the 'timestamp' field directly
data = pd.merge(historical_env_data, historical_wildfire_data, how='left', on='timestamp')

# Create a binary target: if a fire occurred (i.e. 'severity' is present), then 1, else 0.
data['fire_occurred'] = data['severity'].notnull().astype(int)

# Fill missing values (ensuring any NaNs in features are replaced with 0)
data = data.fillna(0)

# Save the merged DataFrame to CSV (optional)
data.to_csv("merged_data.csv", index=False)

features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'vegetation_index', 'human_activity_index']
target = 'fire_occurred'
# ---------------------------
# 4. Check for a Saved Model and Train if Needed
# ---------------------------
model_filename = 'rf_model.pkl'

if os.path.exists(model_filename):
    model = joblib.load(model_filename)
    print("Loaded saved model from", model_filename)
else:
    # ---------------------------
    # 3. Prepare Features and Target for Modeling
    # ---------------------------


    X = data[features]
    y = data[target].astype(int)  # Ensure the target is discrete

    # Compute counts for binary classes and define custom class weights
    count_0 = (y == 0).sum()
    count_1 = (y == 1).sum()
    count_total = count_0 + count_1

    custom_class_weights = {
        0: count_total / count_0 if count_0 > 0 else 1.0,
        1: count_total / count_1 if count_1 > 0 else 1.0,
    }

    print("Custom Class Weights:", custom_class_weights)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to balance the training set
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Train the RandomForestClassifier using the resampled data and custom class weights
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=custom_class_weights)
    model.fit(X_train_res, y_train_res)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save the model for future use
    joblib.dump(model, model_filename)
    print("Model saved to", model_filename)

# ---------------------------
# 5. Future Fire Occurrence Prediction for 2025
# ---------------------------
def predict_future_fire_occurrences(future_environmental_data_file):
    """
    Loads future environmental data for 2025, applies the trained model to predict fire occurrence,
    and optionally generates an interactive map if latitude and longitude columns are present.
    """
    # Load future environmental data with date parsing
    future_env_data = pd.read_csv(future_environmental_data_file, parse_dates=['timestamp'])
    future_env_data = future_env_data.fillna(0)
    future_X = future_env_data[features]

    # Predict fire occurrence using the trained model
    predictions = model.predict(future_X)
    future_env_data['fire_occurred'] = predictions

    # Save the future predictions to a CSV file
    future_env_data.to_csv("future_fire_predictions_2025.csv", index=False)
    print("Future fire predictions saved to 'future_fire_predictions_2025.csv'.")

    # Generate an interactive map if latitude and longitude columns are available
    if 'latitude' in future_env_data.columns and 'longitude' in future_env_data.columns:
        fire_map = folium.Map(location=[46.8139, -71.2082], zoom_start=6)
        for idx, row in future_env_data.iterrows():
            if row['fire_occurred'] == 1:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    fill=True,
                    color='red'
                ).add_to(fire_map)
        fire_map.save('future_fire_map_2025.html')
        print("Interactive fire risk map saved as 'future_fire_map_2025.html'.")

# Example usage: predict future fire occurrences using the 2025 environmental data
future_environmental_data_file = 'future_environmental_data.csv'
predict_future_fire_occurrences(future_environmental_data_file)
