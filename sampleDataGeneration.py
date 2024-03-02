import random
import pandas as pd

# Parameters
x = 8  # Number of airports
y = 3  # Number of aircraft
routes = []

# Generate sample data for routes
# Assuming direct routes between each pair of airports with randomly generated profit and duration
for i in range(1, x + 1):
    for j in range(1, x + 1):
        if i != j:  # No route from an airport to itself
            profit = random.randint(1000, 5000)  # Random profit
            duration = random.randint(1, 24)  # Random duration in hours
            routes.append({'from_airport': i, 'to_airport': j, 'profit': profit, 'duration': duration})

# Convert routes to DataFrame for easier manipulation
routes_df = pd.DataFrame(routes)

# Display the generated sample data
#routes_df.head(10), routes_df.shape

routes_df.to_csv('sample.csv', sep='\t')
