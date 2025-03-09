import cugraph
import cudf

# Create example cudf DataFrames
src = cudf.Series([0, 1, 2, 3, 0, 1, 2])
dst = cudf.Series([1, 2, 3, 0, 2, 3, 0])
weights = cudf.Series([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5])

df = cudf.DataFrame({'src': src, 'dst': dst, 'weights': weights})

# Create a cugraph Graph
G = cugraph.Graph()
G.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weights')

# Specify source and target nodes
source_node = 0
target_node = 3

# Run shortest path algorithm from source
distances, predecessors = cugraph.shortest_path(G, source=source_node)

# Get distance to target node
distance_to_target = distances[target_node]

# Reconstruct the path (optional)
path = []
current_node = target_node
while current_node != source_node:
    path.insert(0, current_node)
    current_node = predecessors[current_node]
path.insert(0, source_node)

print(f"Shortest distance from {source_node} to {target_node}: {distance_to_target}")
print(f"Shortest path from {source_node} to {target_node}: {path}")