import pandas as pd

x = 10  # Number of airports
y = 2

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


# Solve the problem
problem.solve()

# Print the selected routes for each aircraft
for v in problem.variables():
    if v.varValue == 1:
        print(v.name, "=", v.varValue)
