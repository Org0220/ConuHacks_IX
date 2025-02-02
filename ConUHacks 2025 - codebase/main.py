import pandas as pd
from datetime import datetime, timedelta

# Define firefighting resources
resources = {
    "Smoke Jumpers": {
        "deploy_time": timedelta(minutes=30),
        "cost": 5000,
        "units": [datetime.min] * 5
    },
    "Fire Engines": {
        "deploy_time": timedelta(hours=1),
        "cost": 2000,
        "units": [datetime.min] * 10
    },
    "Helicopters": {
        "deploy_time": timedelta(minutes=45),
        "cost": 8000,
        "units": [datetime.min] * 3
    },
    "Tanker Planes": {
        "deploy_time": timedelta(hours=2),
        "cost": 15000,
        "units": [datetime.min] * 2
    },
    "Ground Crews": {
        "deploy_time": timedelta(hours=1, minutes=30),
        "cost": 3000,
        "units": [datetime.min] * 8
    }
}

damage_costs = {
    "low": 50000,
    "medium": 100000,
    "high": 200000
}

severity_rank = {
    "high": 0,
    "medium": 1,
    "low": 2
}

# Load wildfire data
fire_data = pd.read_csv("data/current_wildfiredata.csv")
fire_data['fire_start_time'] = pd.to_datetime(fire_data['fire_start_time'])
fire_data.sort_values(by=['fire_start_time', 'severity'], key=lambda x: x.map(severity_rank), inplace=True)

def assign_fire_unit(fire):
    fire_end_time = fire['fire_start_time']
    # Change these to change the amount of time units are unavailable
    if fire['severity'] == "high":
        fire_duration = timedelta(days=365)  # 2 months or 180 days
    elif fire['severity'] == "medium":
        fire_duration = timedelta(days=365)  # 1 month or 90 days
    else:
        fire_duration = timedelta(days=365)  # 7 days or 30 days
    
    for unit_type, unit_info in sorted(resources.items(), key=lambda x: x[1]['deploy_time']):
        for i in range(len(unit_info['units'])):
            if unit_info['units'][i] <= fire['fire_start_time']:
                unit_info['units'][i] = fire['fire_start_time'] + fire_duration
                return unit_type, unit_info['cost']
    
    return None, damage_costs[fire['severity']]

fire_data['assigned_unit'] = None
fire_data['cost'] = 0

for index, fire in fire_data.iterrows():
    unit, cost = assign_fire_unit(fire)
    fire_data.at[index, 'assigned_unit'] = unit
    fire_data.at[index, 'cost'] = cost

# Sort final dataset by timestamp
fire_data.sort_values(by='fire_start_time', inplace=True)

# Save results
fire_data.to_csv("data/assigned_firefighting_units.csv", index=False)
print("Assignment completed and saved to assigned_firefighting_units.csv")
