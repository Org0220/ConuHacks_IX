import csv
import random
import datetime
import numpy as np

def generate_environmental_data(start_date, end_date, filename):
    current_date = start_date
    environmental_data = []
    
    while current_date <= end_date:
        for hour in range(24):
            timestamp = current_date + datetime.timedelta(hours=hour)
            temperature = round(random.uniform(15.0, 40.0), 1)
            humidity = random.randint(10, 90)
            wind_speed = random.randint(0, 40)
            precipitation = round(random.uniform(0.0, 5.0), 1)
            vegetation_index = random.randint(30, 80)
            human_activity_index = random.randint(0, 100)
            latitude = round(45.0 + random.uniform(-1, 1), 4)
            longitude = round(-73.0 + random.uniform(-1, 1), 4)
            
            environmental_data.append([timestamp, temperature, humidity, wind_speed, precipitation, vegetation_index, human_activity_index, latitude, longitude])
            
        current_date += datetime.timedelta(days=1)
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'temperature', 'humidity', 'wind_speed', 'precipitation', 'vegetation_index', 'human_activity_index', 'latitude', 'longitude'])
        writer.writerows(environmental_data)

def generate_wildfire_data(env_data_filename, wildfire_data_filename, start_year, end_year, fire_probability_threshold=0.25):
    with open(env_data_filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        
        wildfire_data = []
        for row in csv_reader:
            timestamp, temperature, humidity, wind_speed, precipitation, vegetation_index, human_activity_index, latitude, longitude = row
            temperature = float(temperature)
            humidity = int(humidity)
            wind_speed = int(wind_speed)
            precipitation = float(precipitation)
            vegetation_index = int(vegetation_index)
            human_activity_index = int(human_activity_index)
            latitude = float(latitude)
            longitude = float(longitude)
            
            if datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').year < start_year or datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').year > end_year:
                continue

            # Define a probability of fire occurrence based on environmental factors
            normalized_temperature = (temperature - 15) / (40 - 15)
            normalized_humidity = 1 - (humidity / 100)
            normalized_wind_speed = wind_speed / 40
            normalized_precipitation = 1 - (precipitation / 5)
            normalized_vegetation_index = vegetation_index / 80
            normalized_human_activity_index = human_activity_index / 100
            
            fire_probability = (normalized_temperature *
                                normalized_humidity *
                                normalized_wind_speed *
                                normalized_precipitation *
                                normalized_vegetation_index *
                                normalized_human_activity_index)
            
            if fire_probability > fire_probability_threshold:
                fire_start_time = (datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(minutes=random.randint(0, 60))).strftime('%Y-%m-%d %H:%M:%S')
                severity_weights = [(0.5, 'low'), (0.3, 'medium'), (0.2, 'high')]
                severity_choices = [weight[1] for weight in severity_weights]
                severity_probs = [weight[0] for weight in severity_weights]
                severity = np.random.choice(severity_choices, p=severity_probs)
                
                # Use the same location as in the environmental data
                location = f"{latitude},{longitude}"
                
                wildfire_data.append([timestamp, fire_start_time, location, severity, latitude, longitude])
    
    with open(wildfire_data_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'fire_start_time', 'location', 'severity', 'latitude', 'longitude'])
        writer.writerows(wildfire_data)

def generate_all_data():
    # Generate historical environmental data and wildfire data for 2020-2023
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)
    generate_environmental_data(start_date, end_date, 'data/historical_environmental_data.csv')
    generate_wildfire_data('data/historical_environmental_data.csv', 'data/historical_wildfiredata.csv', 2020, 2023)
    
    # Generate current environmental data and wildfire data for 2024
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime(2024, 12, 31)
    generate_environmental_data(start_date, end_date, 'environmental_data.csv')
    generate_wildfire_data('environmental_data.csv', 'current_wildfire_data.csv', 2024, 2024)
    
    # Generate future environmental data for 2025
    start_date = datetime.datetime(2025, 1, 1)
    end_date = datetime.datetime(2025, 12, 31)
    generate_environmental_data(start_date, end_date, 'data/future_environmental_data.csv')

# Generate the data
generate_all_data()