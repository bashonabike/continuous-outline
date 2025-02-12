import math
import numpy as np

import helpers.Enums as Enums
import helpers.TourNode as TourNode

def create_tour_nodes_from_paths(paths, set:Enums.NodeSet):
    """
    Creates a list of linked TourNode objects from a series of paths.

    Args:
        paths: A list of NumPy arrays, where each array represents a path
               and contains the (x, y) coordinates of the nodes.

    Returns:
        A list of lists. The outer list corresponds to the paths, the inner
        list to the TourNodes of that path. Each TourNode has its
        prev/next/angle attributes correctly set.
    """

    all_tour_nodes = []

    for path in paths:
        path_nodes = []
        for i in range(len(path)):
            (x, y) = path[i][0]
            node = TourNode.TourNode(x, y, set)
            path_nodes.append(node)

        # Link nodes and calculate angles within each path:
        for i in range(len(path_nodes)):
            current_node = path_nodes[i]

            if i == 0: path_nodes[i].starter = True
            elif i == len(path_nodes) - 1: path_nodes[i].ender = True

            #TODO: track next angle from prev node, use pi minus it to get pprevangle next node
            if i > 0:  # Set previous node and angle
                current_node.prev = path_nodes[i - 1]
                dx = current_node.prev.x - current_node.x
                dy = current_node.prev.y - current_node.y
                current_node.prev_angle_rad = math.atan2(dy, dx)

            if i < len(path_nodes) - 1:  # Set next node and angle
                current_node.next = path_nodes[i + 1]
                dx = current_node.next.x - current_node.x
                dy = current_node.next.y - current_node.y
                current_node.next_angle_rad = math.atan2(dy, dx)
        all_tour_nodes.extend(path_nodes)
    return all_tour_nodes

def grid_nodes(parsed_nodes, m, n):
    grid = np.empty((n, m), dtype=object)
    for node in parsed_nodes:
        if 0 <= node.x < m and 0 <= node.y < n:  # Check bounds! Important
            grid[node.y, node.x] = node  # Note the order: y, x for indexing

    return grid