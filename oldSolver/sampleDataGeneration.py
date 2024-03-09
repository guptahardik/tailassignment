# import random
# import pandas as pd

# # Parameters
# x = 8  # Number of airports
# y = 3  # Number of aircraft
# routes = []

# # Generate sample data for routes
# # Assuming direct routes between each pair of airports with randomly generated profit and duration
# for i in range(1, x + 1):
#     for j in range(1, x + 1):
#         if i != j:  # No route from an airport to itself
#             profit = random.randint(1000, 5000)  # Random profit
#             duration = random.randint(1, 24)  # Random duration in hours
#             routes.append({'from_airport': i, 'to_airport': j, 'profit': profit, 'duration': duration})

# # Convert routes to DataFrame for easier manipulation
# routes_df = pd.DataFrame(routes)

# # Display the generated sample data
# #routes_df.head(10), routes_df.shape

# routes_df.to_csv('sample.csv', sep='\t')


import random
import pandas as pd

# Parameters
x = 8  # Number of airports
y = 3  # Number of aircraft
routes = []

def calculate_profit(demand):
    base_cost = 500  # Simulated base operational cost
    profit_per_passenger = 50  # Profit per unit of demand after covering base costs
    return max(0, (demand * profit_per_passenger) - base_cost)

# Generate sample data for routes
for i in range(1, x + 1):
    for j in range(1, x + 1):
        if i != j:  # No route from an airport to itself
            demand = random.randint(20, 100)  # Simulate demand
            profit = calculate_profit(demand)
            duration = random.randint(1, 24)  # Random duration in hours
            routes.append({'from_airport': i, 'to_airport': j, 'demand': demand, 'profit': profit, 'duration': duration})

# Convert routes to DataFrame for easier manipulation
routes_df = pd.DataFrame(routes)

# Save the generated routes data with demand and profit to a CSV file
routes_df.to_csv('sample.csv', sep='\t', index=False)

tails = []
for tail_id in range(1, y + 1):
    capacity = random.randint(100, 200)  # Random capacity in a realistic range
    tails.append({'tail_id': tail_id, 'capacity': capacity})

# Convert to DataFrame
tails_df = pd.DataFrame(tails)

# Display the generated tail capacity data
print(tails_df)

# Save to CSV
tails_df.to_csv('tails_capacity.csv', sep='\t', index=False)
