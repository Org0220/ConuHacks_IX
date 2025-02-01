import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import folium

# Load environmental data and historical wildfire data
historical_env_data = pd.read_csv('historical_environmental_data.csv', parse_dates=['timestamp'])
historical_wildfire_data = pd.read_csv('historical_wildfiredata.csv', parse_dates=['timestamp', 'fire_start_time'])

# Merge data on timestamp and create target variable
data = pd.merge(historical_env_data, historical_wildfire_data, how='left', on='timestamp')
data['fire_occurred'] = data['severity'].notnull().astype(int)

# Fill missing values
data = data.fillna(0)

data.to_csv("merged_data.csv")
# Feature selection
features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'vegetation_index', 'human_activity_index']
X = data[features]
y = data['fire_occurred']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict fire occurrences for future environmental data
def predict_fire_occurrences(future_environmental_data_file):
    future_env_data = pd.read_csv(future_environmental_data_file, parse_dates=['timestamp'])
    future_predictions = clf.predict(future_env_data[features])
    future_env_data['fire_risk'] = future_predictions

    # Filter out only rows where fire_risk is 1
    fire_risk_data = future_env_data[future_env_data['fire_risk'] == 1]

    # Print predicted fire risk data to the console
    if not fire_risk_data.empty:
        print(fire_risk_data[['timestamp', 'temperature', 'humidity', 'wind_speed', 'precipitation', 'vegetation_index', 'human_activity_index', 'latitude', 'longitude', 'fire_risk']])



# Example usage
future_environmental_data_file = 'future_environmental_data.csv'
predict_fire_occurrences(future_environmental_data_file)

