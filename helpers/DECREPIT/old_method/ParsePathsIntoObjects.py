import math
import numpy as np

import helpers.Enums as enums
import helpers.DECREPIT.old_method.TourNode as TourNode
import helpers.DECREPIT.old_method.TourConstraints as constr

def create_tour_nodes_from_paths(paths, set:enums.NodeSet):
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

def search_for_cand_next_nodes(bookend_nodes, gridded_nodes):
    for bookend in bookend_nodes:
        exclude = []
        for dist in range(constr.startsearchdist, constr.maxsearchdist + 1, constr.startsearchdist):
            #Form bound box
            lbound, rbound = max(bookend.x - dist, 0), min(bookend.x + dist, gridded_nodes.shape[1] - 1)
            bbound, ubound = max(bookend.y - dist, 0), min(bookend.y + dist, gridded_nodes.shape[0] - 1)
            search_grid = gridded_nodes[bbound:ubound, lbound:rbound]
            all_nodes = search_grid[search_grid != None].flatten().tolist()
            cand_nodes = [c for c in all_nodes if c not in exclude]
            exclude = all_nodes
            bookend.nodes_in_rings_to_oblate.append(cand_nodes)

            #Compute deflection angles
            cand_nodes_formed = []
            for cand in cand_nodes:
                dx = cand.x - bookend.x
                dy = cand.y - bookend.y
                angle = math.atan2(dy, dx)
                dist_sqr = distance_sqr_between_nodes(bookend, cand)
                if bookend.starter:
                    deflect_in = abs(bookend.next.prev_angle_rad - angle)
                else:
                    deflect_in = abs(bookend.prev.next_angle_rad - angle)
                for dir in [enums.Direction.FORWARD, enums.Direction.BACKWARD]:
                    if dir == enums.Direction.FORWARD:
                        if cand.ender: continue
                        deflect_out = abs(cand.next_angle_rad - angle)
                    else:
                        if cand.starter: continue
                        deflect_out = abs(cand.prev_angle_rad - angle)
                    cand_nodes_formed.append({'accum_deflect': deflect_in + deflect_out,
                                              'accum_defl_score': 1.0/(deflect_in + deflect_out + 1.0),
                                              'dist_sqr': dist_sqr,
                                              'net_score': (3.0 if cand.set == enums.NodeSet.OUTER else 1.0)/
                                                           (deflect_in + deflect_out + 0.0001*dist_sqr + 1.0),
                                              'dir': dir,
                                              'node': cand})

            bookend.nodes_in_rings.append(cand_nodes_formed)


def distance_sqr_between_nodes(node1, node2):
        """Calculates the Euclidean distance between two nodes.

        Args:
            node1: A Node object.
            node2: A Node object.

        Returns:
            The Euclidean distance between the two nodes (a float).
        """

        x_diff = node1.x - node2.x
        y_diff = node1.y - node2.y
        # distance = math.isqrt(x_diff ** 2 + y_diff ** 2)  # Or math.sqrt() if you prefer
        return x_diff ** 2 + y_diff ** 2