import pandas as pd
from datetime import datetime, timedelta

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


def main():
    try:
        df = pd.read_csv("current_wildfiredata.csv")
    except Exception as e:
        print("Error loading CSV file:", e)
        return

    df['fire_start_time'] = pd.to_datetime(df['fire_start_time'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['severity_rank'] = df['severity'].str.lower().map(severity_rank)
    df.sort_values(by=["fire_start_time", "severity_rank"], inplace=True)

    addressed_counts = {"low": 0, "medium": 0, "high": 0}
    missed_counts = {"low": 0, "medium": 0, "high": 0}
    total_operational_cost = 0
    total_damage_cost = 0

    for index, row in df.iterrows():
        event_time = row['fire_start_time']
        severity = row['severity'].lower()

        available_options = []
        for res_name, res_info in resources.items():
            for unit_idx, available_time in enumerate(res_info["units"]):
                if event_time >= available_time:
                    available_options.append((res_name, unit_idx, res_info["cost"], res_info["deploy_time"]))

        if available_options:
            chosen_resource = min(available_options, key=lambda x: x[2])
            res_name, unit_idx, cost, deploy_time = chosen_resource

            resources[res_name]["units"][unit_idx] = event_time + deploy_time
            total_operational_cost += cost
            addressed_counts[severity] += 1
        else:
            missed_counts[severity] += 1
            total_damage_cost += damage_costs[severity]

    total_addressed = sum(addressed_counts.values())
    total_missed = sum(missed_counts.values())
    severity_report = {
        "low": addressed_counts["low"] + missed_counts["low"],
        "medium": addressed_counts["medium"] + missed_counts["medium"],
        "high": addressed_counts["high"] + missed_counts["high"]
    }

    print("Number of fires addressed:", total_addressed)
    print("Number of fires delayed:", total_missed)
    print("Total operational costs: $", total_operational_cost)
    print("Estimated damage costs from delayed responses: $", total_damage_cost)
    print("Fire severity report:", severity_report)


if __name__ == "__main__":
    main()