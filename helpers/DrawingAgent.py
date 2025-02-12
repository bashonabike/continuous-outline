#Metrics:
# maximize size of oblit mask
# minimize accum deflection angles
# minimize crossovers
# minimize length of connectors
import helpers.TourConstraints as constr
import helpers.ParsePathsIntoObjects as parse
import helpers.Enums as enums
import helpers.NodeSet as nodeset

import numpy as np
import copy as cp
import random as rd
import cv2

import helpers.AgentStats as stat
class DrawingAgent:
    def __init__(self, dims, node_set:nodeset.NodeSet):
        self.stats = stat.AgentStats()
        self.oblit_mask = np.zeros((dims[0], dims[1]), dtype=np.uint8)
        self.dims = dims
        self.node_set = node_set

        self.tour_path = []


    def tour(self):
        #Pick random outer path to start on
        cur_node = rd.choice(self.node_set.outer_bookends)
        dir = enums.Direction.FORWARD if cur_node.starter else enums.Direction.BACKWARD
        for iter in range(constr.maxiters):
            outbound_angle = self.run_path(cur_node, dir)
            next_node = self.search_for_next_node(cur_node)
            if next_node is None:
                break

            fwd_defl = abs(outbound_angle - next_node.next_angle_rad)
            rev_defl = abs(outbound_angle - next_node.prev_angle_rad)

            if next_node.starter or (not next_node.ender and fwd_defl < rev_defl):
                dir = enums.Direction.FORWARD
                self.stats.accum_defl_rad += fwd_defl
            else:
                dir = enums.Direction.BACKWARD
                self.stats.accum_defl_rad += rev_defl

            self.stats.length_of_connectors += int(round(self.distance_between_nodes(cur_node, next_node),0))

            cur_node = next_node

        #Compute final stats
        self.stats.oblit_mask_size = np.count_nonzero(self.oblit_mask)
        self.stats.crossovers = self.count_crossovers(self.tour_path)
        self.stats.set_final_score()

        #TODO: try some weighting between details and outer paths maybe, for now rely on random


    def run_path(self, cur_node, dir:enums.Direction):
        self.tour_path.append(cur_node)
        if dir == enums.Direction.FORWARD:
            while cur_node.next is not None:
                cv2.line(self.oblit_mask, (cur_node.x, cur_node.y),
                         (cur_node.next.x, cur_node.next.y), (255,255,255), constr.oblatethickness)
                cur_node = cur_node.next
                self.tour_path.append(cur_node)
            return cur_node.prev.next_angle_rad
        else:
            while cur_node.prev is not None:
                cv2.line(self.oblit_mask, (cur_node.x, cur_node.y),
                         (cur_node.prev.x, cur_node.prev.y), (255,255,255), constr.oblatethickness)
                cur_node = cur_node.prev
                self.tour_path.append(cur_node)
            return cur_node.next.prev_angle_rad

    def search_for_next_node(self, cur_node):
        for dist in range(constr.startsearchdist, constr.maxsearchdist + 1, constr.startsearchdist):
            #Form bound box
            lbound, rbound = cur_node.x - dist, cur_node.x + dist
            bbound, ubound = cur_node.y - dist, cur_node.y + dist
            search_grid = self.node_set.gridded_nodes[bbound:ubound, lbound:rbound]

            cand_nodes = [c for c in search_grid[search_grid != None].flatten().tolist() if c.set
                          not in(enums.NodeSet.OUTEROBLIT, enums.NodeSet.DETAILOBLIT)]

            #CHECK FOR OBLATED ONES
            for cand_node in cand_nodes:
                if self.oblit_mask[cand_node.y, cand_node.x] != 0:
                    cand_node.set = enums.NodeSet.OUTEROBLIT if cand_node.set == enums.NodeSet.OUTER \
                        else enums.NodeSet.DETAILOBLIT

            cand_nodes = [c for c in cand_nodes if c.set not in(enums.NodeSet.OUTEROBLIT, enums.NodeSet.DETAILOBLIT)]

            if len(cand_nodes) > 0:
                next_node = rd.choice(cand_nodes)
                return next_node

        return None

    def distance_between_nodes(self, node1, node2):
        """Calculates the Euclidean distance between two nodes.

        Args:
            node1: A Node object.
            node2: A Node object.

        Returns:
            The Euclidean distance between the two nodes (a float).
        """

        x_diff = node1.x - node2.x
        y_diff = node1.y - node2.y
        distance = np.sqrt(x_diff ** 2 + y_diff ** 2)  # Or math.sqrt() if you prefer
        return distance

    def count_crossovers(self, path):
        """
        Counts the number of crossovers in a path defined by a series of nodes.

        Args:
            path: A list or NumPy array of Node objects.

        Returns:
            The number of crossovers (an integer).
        """

        crossover_count = 0
        num_nodes = len(path)

        for i in range(num_nodes - 1):  # Iterate through segments (pairs of consecutive nodes)
            for j in range(i + 2, num_nodes - 1):  # Compare with segments that don't share an endpoint
                # Define line segments
                x1 = path[i].x
                y1 = path[i].y
                x2 = path[i + 1].x
                y2 = path[i + 1].y

                x3 = path[j].x
                y3 = path[j].y
                x4 = path[j + 1].x
                y4 = path[j + 1].y

                # Check for intersection (more robust version)
                denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

                if denominator != 0:  # Lines are not parallel
                    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
                    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

                    if 0 <= ua <= 1 and 0 <= ub <= 1:  # Intersection found
                        crossover_count += 1

        return crossover_count
