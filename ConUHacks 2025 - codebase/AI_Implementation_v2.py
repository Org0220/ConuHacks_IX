import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import folium

MERGED_DATA_PATH = 'data/merged_data.csv'
FIRE_OCCURED_GENERATED_PATH = 'data/future_fire_predictions_2025.csv'
HTML_FILE_PATH = 'data/new_future_fire_map_2025.html'
OUTPUT_FILE_PATH = 'data/new_future_fire_predictions_2025.csv'
SAVED_MODEL_PATH = 'data/rf_severity_model.pkl'

data = pd.read_csv('data/merged_data.csv', parse_dates=['timestamp'])

data_fire = data[data['fire_occurred'] != 0].copy()

data_fire['severity'] = data_fire['severity'].map({'low': 1, 'medium': 2, 'high': 3})

features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'vegetation_index', 'human_activity_index']
target = 'severity'

X = data_fire[features]
y = data_fire[target].astype(int)

count_1 = (y == 1).sum()
count_2 = (y == 2).sum()
count_3 = (y == 3).sum()
count_total = count_1 + count_2 + count_3

custom_class_weights = {
    1: count_total / count_1 if count_1 > 0 else 1.0,
    2: count_total / count_2 if count_2 > 0 else 1.0,
    3: count_total / count_3 if count_3 > 0 else 1.0,
}

print("Custom Class Weights:", custom_class_weights)

model_filename = SAVED_MODEL_PATH

if os.path.exists(SAVED_MODEL_PATH):
    model = joblib.load(SAVED_MODEL_PATH)
    print("Loaded saved severity model from", SAVED_MODEL_PATH)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight=custom_class_weights),
                            param_grid,
                            cv=5,
                            scoring='f1_macro')
    grid_search.fit(X_train_res, y_train_res)
    print("Best Parameters:", grid_search.best_params_)

    model = grid_search.best_estimator_
    
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    joblib.dump(model, SAVED_MODEL_PATH)
    print("Severity model saved to", SAVED_MODEL_PATH)


def predict_future_fire_occurrences():

    future_env_data = pd.read_csv(FIRE_OCCURED_GENERATED_PATH, parse_dates=['timestamp'])
    future_env_data = future_env_data.fillna(0)

    if 'severity' not in future_env_data.columns:
        future_env_data['severity'] = 0

    mask = future_env_data['fire_occurred'] == 1
    if mask.any():
        future_X = future_env_data.loc[mask, features]
        predictions = model.predict(future_X)
        future_env_data.loc[mask, 'severity'] = predictions

    future_env_data.loc[future_env_data['fire_occurred'] == 0, 'severity'] = 0

    future_env_data.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Future fire predictions saved to '{OUTPUT_FILE_PATH}'.")

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
        print("Interactive fire risk map saved as 'new_future_fire_map_2025.html'.")

predict_future_fire_occurrences()
