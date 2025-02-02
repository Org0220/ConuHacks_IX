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

HIST_ENVIREMENTAL_DATA_PATH = 'data/historical_environmental_data.csv'
HIST_WILDFIRE_DATA_PATH = 'data/historical_wildfiredata.csv'
MERGED_DATA_PATH = 'data/merged_data.csv'
FUTURE_ENVIRONMENTAL_DATA_PATH = 'data/future_environmental_data.csv'
HTML_FILE_PATH = 'data/future_fire_map_2025.html'
OUTPUT_FILE_PATH = 'data/future_fire_predictions_2025.csv'
SAVED_MODEL_PATH = 'data/rf_model.pkl'

historical_env_data = pd.read_csv(HIST_ENVIREMENTAL_DATA_PATH, parse_dates=['timestamp'])
historical_wildfire_data = pd.read_csv(HIST_WILDFIRE_DATA_PATH, parse_dates=['timestamp', 'fire_start_time'])

data = pd.merge(historical_env_data, historical_wildfire_data, how='left', on='timestamp')

data['fire_occurred'] = data['severity'].notnull().astype(int)

data = data.fillna(0)

data.to_csv(MERGED_DATA_PATH, index=False)

features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'vegetation_index', 'human_activity_index']
target = 'fire_occurred'
model_filename = SAVED_MODEL_PATH

if os.path.exists(model_filename):
    model = joblib.load(model_filename)
    print("Loaded saved model from", model_filename)
else:


    X = data[features]
    y = data[target].astype(int)

    count_0 = (y == 0).sum()
    count_1 = (y == 1).sum()
    count_total = count_0 + count_1

    custom_class_weights = {
        0: count_total / count_0 if count_0 > 0 else 1.0,
        1: count_total / count_1 if count_1 > 0 else 1.0,
    }

    print("Custom Class Weights:", custom_class_weights)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=custom_class_weights)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(model, model_filename)
    print("Model saved to", model_filename)

def predict_future_fire_occurrences():

    future_env_data = pd.read_csv(FUTURE_ENVIRONMENTAL_DATA_PATH, parse_dates=['timestamp'])
    future_env_data = future_env_data.fillna(0)
    future_X = future_env_data[features]

    predictions = model.predict(future_X)
    future_env_data['fire_occurred'] = predictions

    future_env_data.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Future fire predictions saved to {OUTPUT_FILE_PATH}.")

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
        fire_map.save(HTML_FILE_PATH)
        print(f"Interactive fire risk map saved as {HTML_FILE_PATH}.")

predict_future_fire_occurrences()
