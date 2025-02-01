import csv
import datetime

# A model solution for part one
class FireResource:
    def __init__(self, resource_type, deployment_time, cost, total_units):
        self.resource_type = resource_type
        self.deployment_time = deployment_time
        self.cost = cost
        self.total_units = total_units
        self.available_units = total_units

    def deploy(self):
        if self.available_units > 0:
            self.available_units -= 1
            return True
        return False

    def reset(self):
        self.available_units = self.total_units

smoke_jumpers = FireResource("Smoke Jumpers", datetime.timedelta(minutes=30), 5000, 5)
fire_engines = FireResource("Fire Engines", datetime.timedelta(hours=1), 2000, 10)
helicopters = FireResource("Helicopters", datetime.timedelta(minutes=45), 8000, 3)
tanker_planes = FireResource("Tanker Planes", datetime.timedelta(hours=2), 15000, 2)
ground_crews = FireResource("Ground Crews", datetime.timedelta(hours=1, minutes=30), 3000, 8)

resources = [smoke_jumpers, fire_engines, helicopters, tanker_planes, ground_crews]

def read_wildfire_data(filename):
    with open(filename, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        wildfire_data = []
        for row in csv_reader:
            timestamp = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            fire_start_time = datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
            location = row[2]
            severity = row[3]
            wildfire_data.append((timestamp, fire_start_time, severity))
        return wildfire_data

def deploy_resources(wildfire_data):
    wildfire_data.sort(key=lambda x: (x[1], x[2]), reverse=False)  # Sort primarily by fire start time, then severity
    total_cost = 0
    delayed_responses = {"low": 0, "medium": 0, "high": 0}
    delayed_costs = {"low": 50000, "medium": 100000, "high": 200000}
    actual_delayed_cost = 0
    report = {"low": 0, "medium": 0, "high": 0}

    severity_weight = {"low": 1, "medium": 2, "high": 3}  # Assigning weights to the severities

    for wildfire in wildfire_data:
        severity = wildfire[2]
        min_cost = float('inf')
        best_resource = None

        for resource in resources:
            if resource.available_units > 0:
                # Calculate potential combined cost: operating cost + weighted potential damage cost
                potential_cost = resource.cost + delayed_costs[severity] / severity_weight[severity]
                if potential_cost < min_cost:
                    min_cost = potential_cost
                    best_resource = resource

        if best_resource and best_resource.deploy():
            total_cost += best_resource.cost
            report[severity] += 1
        else:
            delayed_responses[severity] += 1
            actual_delayed_cost += delayed_costs[severity]

    return total_cost, delayed_responses, actual_delayed_cost, report

def reset_resources():
    for resource in resources:
        resource.reset()

def main(current_wildfire_data_file):
    wildfire_data = read_wildfire_data(current_wildfire_data_file)
    total_cost, delayed_responses, actual_delayed_cost, report = deploy_resources(wildfire_data)
    reset_resources()

    print(f"Number of fires addressed: {sum(report.values())}")
    print(f"Number of fires delayed: {sum(delayed_responses.values())}")
    print(f"Total operational costs: ${total_cost}")
    print(f"Estimated damage costs from delayed responses: ${actual_delayed_cost}")
    print(f"Fire severity report: {report}")

# Example usage
current_wildfire_data_file = 'current_wildfiredata.csv'
main(current_wildfire_data_file)