import networkx as nx
import heapq
import pandas as pd
import pickle

def yen_algorithm(graph, source, target, k):
    """Yen's algorithm to find K-shortest paths from source to target in a graph."""
    A = []  # List to store the K-shortest paths
    B = []  # List to store potential K-shortest paths

    # Compute the shortest path from source to target
    shortest_path = nx.shortest_path_length(graph, source=source, target=target, weight='weight')
    
    A.append(nx.shortest_path(graph, source=source, target=target, weight='weight'))

    for kth in range(1, k):
        for i in range(len(A[-1]) - 1): 
            spur_node = A[-1][i]
            root_path = A[-1][:i + 1]

            graph.remove_edge(root_path[-1], A[-1][i + 1])

            try:
                spur_path_length = nx.shortest_path_length(graph, source=spur_node, target=target, weight='weight')
                total_path = nx.shortest_path(graph, source=source, target=spur_node, weight='weight')[:-1] + \
                             nx.shortest_path(graph, source=spur_node, target=target, weight='weight')
                total_path_length = nx.shortest_path_length(graph, source=source, target=target, weight='weight')
                B.append((total_path, total_path_length))
            except nx.NetworkXNoPath:
                continue

            graph.add_edge(root_path[-1], A[-1][i + 1])

        if not B:
            break

        B.sort(key=lambda x: x[1])
        thisone = B.pop(0)[0]
        if thisone not in A:
            A.append(thisone)

    return A

# Create a directed graph
with open('/Users/robmailley/Documents/103Final/data/flights.pkl', 'rb') as f:
    flights_fare = pickle.load(f)

graph = nx.DiGraph()
graph.add_weighted_edges_from(flight_list)

# Define source and target nodes
source = 'BDL'
target = 'PVU'

# Number of shortest paths to find
k = 2

# Find K-shortest paths using Yen's algorithm
myalg = yen_algorithm(graph, source, target, k)
#k_shortest_path = k_shortest_paths(graph, source, target, k)

# Print the K-shortest paths
print(f"K-shortest paths from {source} to {target}:")
for idx, path in enumerate(myalg):
    print(f"Path {idx + 1}: {path}")
