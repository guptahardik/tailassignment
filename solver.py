import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


x = 8  # Number of airports
y = 2

#aircraft_demand = {1: 2, 2: 1, 3: 1, 4: 2, 5: 1}  # Example demand


# Load the data
# Load the data with tab as the separator
data = pd.read_csv('sample.csv', sep='\t')

# After loading, it's a good idea to print the column names again to confirm they're now correctly recognized
print(data.columns)

from pulp import *

# Define the problem
problem = LpProblem("Aircraft_Route_Scheduling", LpMaximize)

#Decision
x = LpVariable.dicts("route", 
                     ((i, j, k) for i in data.from_airport.unique() for j in data.to_airport.unique() for k in range(1, y+1)), 
                     cat='Binary')


# Objective function
problem += lpSum([data.iloc[i].profit * x[(data.iloc[i].from_airport, data.iloc[i].to_airport, k)] 
                  for i in range(len(data)) for k in range(1, y+1)])


# Aircraft utilization constraint
for k in range(1, y+1):
    problem += lpSum([data.iloc[i].duration * x[(data.iloc[i].from_airport, data.iloc[i].to_airport, k)] 
                      for i in range(len(data))]) <= 24, f"Aircraft_{k}_utilization"

# Ensure a route is assigned to at most one aircraft
for i in range(len(data)):
    problem += lpSum([x[(data.iloc[i].from_airport, data.iloc[i].to_airport, k)] for k in range(1, y+1)]) <= 1, f"Route_{i}_assignment"

# for airport_id in aircraft_demand.keys():
#     # Calculate the net number of aircraft at each airport at the end of the day
#     aircraft_arriving = lpSum([x[(i, j, k)] for i, j, k in x.keys() if j == airport_id])
#     aircraft_departing = lpSum([x[(i, j, k)] for i, j, k in x.keys() if i == airport_id])
    
#     # Ensure net aircraft meets or exceeds the next day's demand
#     problem += (aircraft_arriving - aircraft_departing >= aircraft_demand[airport_id], f"NextDayDemand_{airport_id}")

# Assuming `data` contains all possible routes
# For each aircraft k
for k in range(1, y+1):
    # For each airport i
    for i in data.from_airport.unique():
        # Sum of departures from airport i by aircraft k
        departures = lpSum(x[(i, j, k)] for j in data.to_airport.unique() if (i, j, k) in x)
        
        # Sum of arrivals to airport i by aircraft k
        arrivals = lpSum(x[(j, i, k)] for j in data.from_airport.unique() if (j, i, k) in x)
        
        # Continuity constraint: departures must equal arrivals for each aircraft at each airport
        problem += (departures == arrivals, f"Continuity_aircraft_{k}_airport_{i}")


# Solve the problem
problem.solve()

# Print the selected routes for each aircraft
for v in problem.variables():
    if v.varValue == 1:
        print(v.name, "=", v.varValue)


#selected_routes = [v.name for v in problem.variables() if v.varValue == 1]

# Parse the selected routes to extract the route and tail information
selected_routes = [v.name for v in problem.variables() if v.varValue == 1]

# New parsing logic
routes_with_tails = []
for route in selected_routes:
    # Remove 'route_(' prefix and ')' suffix, then split by ',_'
    parts = route.replace("route_(", "").replace(")", "").split(",_")
    from_airport = int(parts[0])
    to_airport = int(parts[1])
    tail = int(parts[2])
    routes_with_tails.append((from_airport, to_airport, tail))

# Create a directed graph
G = nx.DiGraph()

# Add edges with tails as labels
for from_airport, to_airport, tail in routes_with_tails:
    G.add_edge(from_airport, to_airport, tail=f"Tail {tail}")

# Draw the graph
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)  # for consistent layout
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=20, font_weight="bold", arrows=True)
edge_labels = nx.get_edge_attributes(G, 'tail')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Optimal Aircraft Routes with Tail Assignments")
plt.show()

# Assuming `data` contains columns 'from_airport' and 'to_airport' representing all possible routes
all_airports = set(data['from_airport'].unique()).union(set(data['to_airport'].unique()))

# Create a directed graph
G = nx.DiGraph()

# Add all airports as nodes
G.add_nodes_from(all_airports)

# Add edges with tails as labels, based on the selected routes
for from_airport, to_airport, tail in routes_with_tails:
    G.add_edge(from_airport, to_airport, tail=f"Tail {tail}")

# Draw the graph
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)  # Use spring layout for positioning nodes
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=20, font_weight="bold", arrows=True)
edge_labels = nx.get_edge_attributes(G, 'tail')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Optimal Aircraft Routes with Tail Assignments")
plt.show()

N = len(all_airports)

# Create a mapping of airport IDs to matrix indices
airport_to_index = {airport: idx for idx, airport in enumerate(sorted(all_airports))}

# Initialize the adjacency matrix with zeros
adjacency_matrix = np.zeros((N, N))

# Fill the matrix based on selected routes
for from_airport, to_airport, _ in routes_with_tails:
    from_idx = airport_to_index[from_airport]
    to_idx = airport_to_index[to_airport]
    adjacency_matrix[from_idx, to_idx] = 1  # Or use the tail number if you prefer

# Display the adjacency matrix
print(adjacency_matrix)