from collections import deque

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import helpers.mazify.temp_options as options

def build_grid_net(grid):
    """
    Args:
        grid: 2D NumPy ndarray representing the grid (weights).
    """
    rows, cols = grid.shape
    graph = nx.DiGraph()  # Directed graph for weighted edges

    # Add nodes and edges
    for r in range(rows):
        for c in range(cols):
            graph.add_node((r, c))
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != options.dumb_node_blank_weight:
                        neighbors.append((nr, nc))
            for neighbor in neighbors:
                graph.add_edge((r, c), neighbor, weight=grid[neighbor])

    return graph

def build_sparse_grid_net(grid):
    """
    Args:
        grid: 2D NumPy ndarray representing the grid (weights).
    """
    rows, cols = grid.shape
    num_nodes = rows * cols

    # 1. Construct Sparse Graph
    graph_sparse = np.zeros((num_nodes, num_nodes))

    def get_node_index(row, col):
        return row * cols + col

    for r in range(rows):
        for c in range(cols):
            current_node = get_node_index(r, c)
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append((nr, nc))
            for neighbor in neighbors:
                neighbor_node = get_node_index(neighbor[0], neighbor[1])
                graph_sparse[current_node, neighbor_node] = grid[
                    neighbor]  # weight of the edge is the weight of the neighbor.

    graph_sparse = csr_matrix(graph_sparse)

    return graph_sparse

def networkx_weighted_grid_path(graph, start, end):
    """
    Finds the shortest path in a weighted grid using NetworkX.

    Args:
        graph: nbetworkx graph representing the weighted grid.
        start: Tuple (row, col) representing the start point.
        end: Tuple (row, col) representing the end point.

    Returns:
        List of tuples representing the shortest path, or None if no path is found.
    """

    try:
        path = nx.shortest_path(graph, start, end, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return None

def sparse_weighted_grid_path_lengths(grid, graph_sparse):

    rows, cols = grid.shape
    def get_node_index(row, col):
        return row * cols + col
    # 2. Run Dijkstra's Algorithm with predecessors
    distances, predecessors = dijkstra(csgraph=graph_sparse, directed=False,return_predecessors=True)
    distances = distances.astype(np.uint16)

    # Reconstruct the path
    shortest_paths = np.zeros_like(predecessors, dtype=list)
    for i in range(len(shortest_paths)):
        for j in range(len(shortest_paths)):
            if i == j:
                shortest_paths[i][j] = [(i, j)]
            elif j < i:
                #Mirror if can halve efforts
                shortest_paths[i][j] = shortest_paths[j][i][::-1]
            else:
                path = []
                current_node = j
                while current_node != i:
                    path.insert(0, (current_node // cols, current_node % cols))  # Convert node index back to (row, col)
                    current_node = predecessors[i][current_node]
                path.insert(0, (i // cols, i % cols))  # add start node to path.
                shortest_paths[i][j] = path

    return distances, shortest_paths


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()} miles")
    index = routing.Start(0)
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    route_indexes = []
    while not routing.IsEnd(index):
        plan_output += f" {manager.IndexToNode(index)} ->"
        route_indexes.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += f" {manager.IndexToNode(index)}\n"
    plan_output += f"Route distance: {route_distance}miles\n"
    print(plan_output)
    return route_indexes



def find_shortest_path_with_coverage(weighted_grid, required_nodes_grid, required_nodes_list):
    """
    Finds the shortest path on a grid that covers at least 90% of required nodes.

    Args:
        grid: 2D list representing the grid, where each cell is a list of nodes.
        required_nodes: List of required node IDs.

    Returns:
        List of node IDs representing the shortest path.
    """
    #Build distance matrix
    required_node_coords_flat = []
    rows = len(required_nodes_grid)
    m, n = required_nodes_grid.shape

    for y in range(m):
        for x in range(n):
            if required_nodes_grid[y][x]:  # Check if the element is not empty
                required_node_coords_flat.append((y, x, required_nodes_grid[y][x]))
    required_node_coords_flat.sort(key=lambda x: x[2])

    #Build graph & full dist matrix
    # graph = build_grid_net(weighted_grid)
    graph_sparse =build_sparse_grid_net(weighted_grid)
    distance_matrix, distance_shortest_path_matrix = sparse_weighted_grid_path_lengths(weighted_grid, graph_sparse)

    #Build distance matrix from required nodes
    req_nodes_muxed = [r[0]*n + r[1] for r in required_node_coords_flat]
    req_nodes_muxed.sort()

    def extract_subset(array_2d, rows, cols):
        """
        Creates a copy of a 2D NumPy ndarray containing only specific rows and columns.

        Args:
            array_2d: The input 2D NumPy ndarray.
            rows: List or NumPy array of row indices to include.
            cols: List or NumPy array of column indices to include.

        Returns:
            A new 2D NumPy ndarray containing the selected rows and columns.
        """

        return array_2d[rows, :][:, cols]

    distance_matrix_req = extract_subset(distance_matrix, req_nodes_muxed, req_nodes_muxed).tolist()
    distance_shortest_path_matrix = extract_subset(distance_shortest_path_matrix, req_nodes_muxed,
                                                   req_nodes_muxed).tolist()

    #Build distance matrix
    # distance_matrix = [[999999 for _ in range(len(required_node_coords_flat))] for _ in
    #                                range(len(required_node_coords_flat))]
    # distance_shortest_path_matrix = [[None for _ in range(len(required_node_coords_flat))] for _ in
    #                                range(len(required_node_coords_flat))]

    # for i in range(len(distance_matrix)):
    #     for j in range(len(distance_matrix)):
    #         if i == j:
    #             distance_shortest_path_matrix[i][j] = [(i, j)]
    #             distance_matrix[i][j] = 0
    #         elif j < i:
    #             #Mirror if can halve efforts
    #             distance_shortest_path_matrix[i][j] = distance_shortest_path_matrix[j][i]
    #             distance_matrix[i][j] = distance_matrix[j][i]
    #         else:
    #             shortest_path = networkx_weighted_grid_path(graph,
    #                                                         required_node_coords_flat[i][0:2],
    #                                                         required_node_coords_flat[j][0:2])
    #             sparse_weighted_grid_path_length(weighted_grid, graph_sparse, required_node_coords_flat[i][0:2],
    #                                              required_node_coords_flat[j][0:2])
    #             if shortest_path is not None:
    #                 distance_shortest_path_matrix[i][j] = shortest_path
    #                 total_length = 0
    #                 for i in range(len(shortest_path) - 1):
    #                     total_length += weighted_grid[shortest_path[i + 1]]
    #                 distance_matrix[i][j] = total_length

    data = {}
    data["distance_matrix"] = distance_matrix_req
    data["num_vehicles"] = 1
    data["depot"] = 0

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    route_indexes = []
    if solution:
        route_indexes = print_solution(manager, routing, solution)
    coords = [required_node_coords_flat[i][0:2] for i in route_indexes]
    y_coords, x_coords = zip(*coords)  # Unzip the coordinates

    plt.plot(x_coords, y_coords, marker='o')  # Plot the line with markers
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Line Plot of Coordinates")
    plt.grid(True)
    plt.show()


    sfsd=""

    #
    # m, n = len(grid), len(grid[0])
    # required_nodes = set(required_nodes)
    # num_required = len(required_nodes)
    # coverage_threshold = int(0.4 * num_required)
    #
    # if num_required == 0:
    #     return []
    #
    # start_nodes = []
    # for r in range(m):
    #     for c in range(n):
    #         if grid[r][c]:
    #             start_nodes.extend(grid[r][c])
    #
    # if not start_nodes:
    #     return []
    #
    # queue = deque([(start_node, [start_node]) for start_node in start_nodes])
    # visited = set(start_nodes) #SET THIS TO EMPTY???
    #
    # while queue:
    #     current_node, path = queue.popleft()
    #     r, c = -1, -1
    #
    #     # Find grid location of current_node
    #     for i in range(m):
    #         for j in range(n):
    #             if current_node in grid[i][j]:
    #                 r, c = i, j
    #                 break
    #         if r != -1:
    #             break
    #
    #     covered_required = set(path).intersection(required_nodes)
    #     if len(covered_required) >= coverage_threshold:
    #         return path
    #
    #     neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1),
    #                  (r - 1, c - 1), (r + 1, c + 1), (r + 1, c - 1), (r - 1, c + 1)]
    #     for nr, nc in neighbors:
    #         if 0 <= nr < m and 0 <= nc < n and grid[nr][nc]:
    #             for neighbor_node in grid[nr][nc]:
    #                 if neighbor_node not in visited:
    #                     visited.add(neighbor_node)
    #                     queue.append((neighbor_node, path + [neighbor_node]))
    #
    # return []  # No path found

# # Example Usage:
# grid = [
#     [[1, 2], [], [3]],
#     [[], [4, 5], []],
#     [[6], [], [7, 8]]
# ]
# required_nodes = [1, 3, 5, 7]
#
# path = find_shortest_path_with_coverage(grid, required_nodes)
# print("Shortest path covering 90%:", path)
#
# grid2 = [
#     [[1, 2]],
#     [[3, 4]]
# ]
# required_nodes2 = [1, 4]
# path2 = find_shortest_path_with_coverage(grid2, required_nodes2)
# print("Shortest path covering 90%:", path2)
#
# grid3 = [
#     [[]]
# ]
# required_nodes3 = [1]
# path3 = find_shortest_path_with_coverage(grid3, required_nodes3)
# print("Shortest path covering 90%:", path3)
#
# grid4 = [
#     [[1],[2]],
#     [[3],[4]]
# ]
# required_nodes4 = [1,4]
# path4 = find_shortest_path_with_coverage(grid4, required_nodes4)
# print("Shortest path covering 90%:", path4)