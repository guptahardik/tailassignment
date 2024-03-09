import pandas as pd
from pulp import *
import networkx as nx
import matplotlib.pyplot as plt

def parse_variable_string(variable_string):
    split_string = variable_string.split('(')[1].split(')')[0]  # Get the content within the first parentheses
    i, j, val = split_string.split(',_')  # Split by ',_' to separate the elements
    i = i.replace("'", "")  # Remove single quotes from the origin airport code
    j = j.replace("'", "")  # Remove single quotes from the destination airport code
    val = val.strip()  # Strip any leading/trailing whitespace from the aircraft ID
    return i, j, val


routes_df = pd.read_csv('routes.csv')
aircrafts_df = pd.read_csv('aircrafts.csv')
aircrafts_df.dropna(subset=['numSeats'], inplace=True)


opcost = 0.5  # Operating cost per mile
speed = 400


duration_dict = {}


for index, route in routes_df.iterrows():
    origin = route['Origin']
    dest = route['Dest']
    distance = route['Distance']
    
    duration = distance / speed
    
    if origin not in duration_dict:
        duration_dict[origin] = {}
    
    duration_dict[origin][dest] = int(duration)

profit_dict = {}

for index, route in routes_df.iterrows():
    origin = route['Origin']
    dest = route['Dest']
    fare = route['Fare']
    distance = route['Distance']
    
    if origin not in profit_dict:
        profit_dict[origin] = {}
    if dest not in profit_dict[origin]:
        profit_dict[origin][dest] = {}
    
    for _, aircraft in aircrafts_df.iterrows():
        aircraft_id = aircraft['id']
        numSeats = aircraft['numSeats']
        
        profit = fare * numSeats - opcost * distance
        
        profit_dict[origin][dest][aircraft_id] = profit
 
# Optimization model
model = LpProblem("Aircraft_Route_Scheduling", LpMaximize)

# Decision variables
x = LpVariable.dicts("route_aircraft_assignment",
                     ((i, j, k) for i in profit_dict for j in profit_dict[i] for k in profit_dict[i][j]),
                     cat='Binary')

# Objective function: Maximize total profit
model += lpSum([profit_dict[i][j][k] * x[(i, j, k)] for i in profit_dict for j in profit_dict[i] for k in profit_dict[i][j]])

# Constraints
# Each aircraft's total flying time does not exceed 24 hours
for k in aircrafts_df['id']:
    model += lpSum([duration_dict[i][j] * x[(i, j, k)] for i in duration_dict for j in duration_dict[i] if k in profit_dict[i][j]]) <= 24

# At most one aircraft per route
for i in duration_dict:
    for j in duration_dict[i]:
        model += lpSum([x[(i, j, k)] for k in profit_dict[i][j]]) <= 1

# Each aircraft must operate a minimum of 8 hours
min_duration = 8
for k in aircrafts_df['id']:
    model += lpSum([duration_dict[i][j] * x[(i, j, k)] for i in duration_dict for j in duration_dict[i] if k in profit_dict[i][j]]) >= min_duration


# Adjusted continuity constraint for correct iteration and indexing
base_airports = ['PHX', 'HSV, JFK']  # Your base airports

# Update model constraints
for k in aircrafts_df['id']:
    for airport in set(routes_df['Origin'].tolist() + routes_df['Dest'].tolist()):
        departures = lpSum([x[(i, j, k)] for i in duration_dict for j in duration_dict[i] if i == airport and k in profit_dict[i][j]])
        arrivals = lpSum([x[(i, j, k)] for i in duration_dict for j in duration_dict[i] if j == airport and k in profit_dict[i][j]])
        
        # For non-base airports, enforce arrivals to equal departures to ensure continuity
        if airport not in base_airports:
            model += (departures == arrivals, f"Continuity_{airport}_{k}")
        else:
            # For base airports, implement specific logic as per your operational requirements
            # This could involve ensuring a minimum number of arrivals/departures
            # or other constraints to manage the beginning and end of the sequence.
            model += (departures >= 1, f"DepartureFromBase_{airport}_{k}")
            model += (arrivals >= 1, f"ArrivalToBase_{airport}_{k}")

model.solve()


all_parsed_variables = []

for v in model.variables():
    # Extract the variable name without the 'route_' prefix and split it

    i, j, val = parse_variable_string(v.name)
    all_parsed_variables.append([i,j, val, v.varValue])




routes_with_tails = [[i, j, id_] for i, j, id_, value in all_parsed_variables if value == 1]

G = nx.DiGraph()


for origin, destination, tail_number in routes_with_tails:
    G.add_edge(origin, destination, tail=tail_number)


plt.figure(figsize=(12, 10))
pos = nx.kamada_kawai_layout(G)  # Positions for all nodes


nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=14, font_weight="bold", edge_color="gray", width=2)


edge_labels = nx.get_edge_attributes(G, 'tail')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

plt.title("Optimal Aircraft Routes with Tail Assignments", fontsize=16)
plt.axis('off')
plt.show()