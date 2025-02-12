#Metrics:
# maximize size of oblit mask
# minimize accum deflection angles
# minimize crossovers
# minimize length of connectors
import helpers.TourConstraints as constr
import helpers.ParsePathsIntoObjects as parse
import helpers.Enums as enums
import helpers.NodeSet as nodeset
import helpers.AgentStats as stat
import helpers.TourNode as tournode

import numpy as np
import copy as cp
import random as rd
import cv2
import time
from scipy.ndimage import label
import math

class DrawingAgent:
    def __init__(self, dims, node_set:nodeset.NodeSet):
        self.stats = stat.AgentStats()
        self.blank = np.zeros((dims[0], dims[1]), dtype=np.uint8)
        self.oblit_mask = self.blank.copy()
        self.oblit_mask_outer = self.blank.copy()
        self.crowding_mask = self.blank.copy()

        self.dims = dims
        self.node_set = node_set

        self.tour_path = []


    def tour(self):
        #Pick random outer path to start on
        startbroad = time.perf_counter_ns() // 1000
        # cur_node = rd.choice(self.node_set.outer_bookends)
        cur_node = rd.choice(self.node_set.bookends)
        dir = enums.Direction.FORWARD if cur_node.starter else enums.Direction.BACKWARD
        loopback = False
        for iter in range(constr.maxiters):
            start = time.perf_counter_ns() // 1000
            cur_node, outbound_angle = self.run_path(cur_node, dir, loopback)
            # cv2.imshow("oblit_mask", self.oblit_mask)
            # cv2.imshow("oblit_nodes_mask", self.oblit_nodes_mask)
            # cv2.waitKey(0)
            end = time.perf_counter_ns() // 1000
            self.stats.pathtime += float(end - start)/1000.0

            start = time.perf_counter_ns() // 1000
            next_node, next_dir = self.search_for_next_node(cur_node)
            end = time.perf_counter_ns() // 1000
            self.stats.searchtime += float(end - start)/1000.0
            if next_node is None:
                #If already tried both ends of path, doneskies
                if loopback: break

                #Try to loop back
                loopback = True
                fwd_defl = rev_defl = math.pi
                next_node = cur_node
            else:
                loopback = False
                fwd_defl = abs(outbound_angle - next_node.next_angle_rad)
                rev_defl = abs(outbound_angle - next_node.prev_angle_rad)

            if (next_node.starter or (not next_node.ender and next_dir == enums.Direction.FORWARD) or
                    (next_dir is None and not next_node.ender and fwd_defl < rev_defl)):
                dir = enums.Direction.FORWARD
                self.stats.accum_defl_rad += fwd_defl
            else:
                dir = enums.Direction.BACKWARD
                self.stats.accum_defl_rad += rev_defl

            self.stats.length_of_connectors += self.distance_between_nodes(cur_node, next_node)

            cur_node = next_node

        endbroad = time.perf_counter_ns() // 1000
        # print("pathfindingtot: "+ str(float(endbroad-startbroad)/1000.0))
        #Compute final stats
        self.stats.oblit_mask_size = np.count_nonzero(self.oblit_mask)
        self.stats.oblit_mask_outer_size = np.count_nonzero(self.oblit_mask_outer)
        start = time.perf_counter_ns() // 1000
        self.count_crowdings()
        end = time.perf_counter_ns() // 1000
        # print("crossovers: "+ str(float(end-start)/1000.0))
        self.stats.set_final_score()
        # print("pathtime: " + str(self.stats.pathtime))
        # print("searchtime: " + str(self.stats.searchtime))
        # print("oblatedrawtime: " + str(self.stats.oblatedrawtime))
        # print("oblatemaskime: " + str(self.stats.oblatemaskime))
        # print("oblatefindpixelstime: " + str(self.stats.oblatefindpixelstime))
        # print("oblateiternodestime: " + str(self.stats.oblateiternodestime))

        #TODO: try some weighting between details and outer paths maybe, for now rely on random


    def run_path(self, cur_node, dir:enums.Direction, loopback=False):
        if not loopback:self.tour_path.append(cur_node)
        if dir == enums.Direction.FORWARD:
            while cur_node.next is not None:
                cv2.line(self.oblit_mask, (cur_node.x, cur_node.y),
                         (cur_node.next.x, cur_node.next.y), (255,255,255), constr.oblatethickness)
                cv2.line(self.crowding_mask, (cur_node.x, cur_node.y),
                         (cur_node.next.x, cur_node.next.y), (255,255,255), constr.crowdingthickness)
                if cur_node.set in (enums.NodeSet.OUTER, enums.NodeSet.OUTEROBLIT):
                    cv2.line(self.oblit_mask_outer, (cur_node.x, cur_node.y),
                             (cur_node.next.x, cur_node.next.y), (255, 255, 255), constr.oblatethickness)
                # self.oblate_pixels_on_line(cur_node, cur_node.next)
                cur_node = cur_node.next
                self.tour_path.append(cur_node)
            return cur_node, cur_node.prev.next_angle_rad
        else:
            while cur_node.prev is not None:
                cv2.line(self.oblit_mask, (cur_node.x, cur_node.y),
                         (cur_node.prev.x, cur_node.prev.y), (255,255,255), constr.oblatethickness)
                cv2.line(self.crowding_mask, (cur_node.x, cur_node.y),
                         (cur_node.prev.x, cur_node.prev.y), (255,255,255), constr.crowdingthickness)
                if cur_node.set in (enums.NodeSet.OUTER, enums.NodeSet.OUTEROBLIT):
                    cv2.line(self.oblit_mask_outer, (cur_node.x, cur_node.y),
                             (cur_node.prev.x, cur_node.prev.y), (255, 255, 255), constr.oblatethickness)
                # self.oblate_pixels_on_line(cur_node, cur_node.prev)
                cur_node = cur_node.prev
                self.tour_path.append(cur_node)
            return cur_node, cur_node.next.prev_angle_rad

    def oblate_pixels_on_line(self, start_node, end_node):
        #SLOWWWWW MAYBE STICK WITH INLINE COPMP
        start = time.perf_counter_ns() // 1000
        cur_blank = self.blank.copy()
        cv2.line(cur_blank, (start_node.x, start_node.y),
                 (end_node.x, end_node.y), (255,255,255), constr.oblatethickness)
        end = time.perf_counter_ns() // 1000
        self.stats.oblatedrawtime += float(end - start)/1000.0


        start = time.perf_counter_ns() // 1000
        mask = cur_blank.astype(bool)
        masked = self.node_set.gridded_nodes[mask]
        end = time.perf_counter_ns() // 1000
        self.stats.oblatemaskime += float(end - start)/1000.0

        start = time.perf_counter_ns() // 1000

        oblated_nodes = masked[masked != None].flatten().tolist()
        end = time.perf_counter_ns() // 1000
        self.stats.oblatefindpixelstime += float(end - start)/1000.0

        start = time.perf_counter_ns() // 1000

        for node in oblated_nodes:
            if node.set == enums.NodeSet.OUTER:  node.set = enums.NodeSet.OUTEROBLIT
            elif node.set == enums.NodeSet.DETAIL:  node.set = enums.NodeSet.DETAILOBLIT
        end = time.perf_counter_ns() // 1000
        self.stats.oblateiternodestime += float(end - start)/1000.0



    def search_for_next_node(self, cur_node:tournode.TourNode):
        for oblates, searches in zip(cur_node.nodes_in_rings_to_oblate, cur_node.nodes_in_rings):
            cand_to_check = [c for c in oblates if c.set
                          not in (enums.NodeSet.OUTEROBLIT, enums.NodeSet.DETAILOBLIT)]

            # CHECK FOR OBLATED ONES
            for cand_node in cand_to_check:
                if self.oblit_mask[cand_node.y, cand_node.x] != 0:
                    cand_node.set = enums.NodeSet.OUTEROBLIT if cand_node.set == enums.NodeSet.OUTER \
                        else enums.NodeSet.DETAILOBLIT
                if self.crowding_mask[cand_node.y, cand_node.x] != 0:
                    cand_node.crowded = True


            cand_nodes_set = [c for c in searches if c['node'].set not in (enums.NodeSet.OUTEROBLIT, enums.NodeSet.DETAILOBLIT)]

            if len(cand_nodes_set) > 0:
                next_node = rd.choices(cand_nodes_set, weights=[n['net_score']/(1.0 if not n['node'].crowded else constr.crowdlimiter)
                                                                for n in cand_nodes_set], k=1)[0]
                return next_node['node'], next_node['dir']

        return None, None

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
        distance = math.isqrt(x_diff ** 2 + y_diff ** 2)  # Or math.sqrt() if you prefer
        return distance

    # def count_crossovers(self, path):
    #     """
    #     Counts the number of crossovers in a path defined by a series of nodes.
    #
    #     Args:
    #         path: A list or NumPy array of Node objects.
    #
    #     Returns:
    #         The number of crossovers (an integer).
    #     """
    #
    #     crossover_count = 0
    #     num_nodes = len(path)
    #
    #     for i in range(num_nodes - 1):  # Iterate through segments (pairs of consecutive nodes)
    #         for j in range(i + 2, num_nodes - 1):  # Compare with segments that don't share an endpoint
    #             # Define line segments
    #             x1 = path[i].x
    #             y1 = path[i].y
    #             x2 = path[i + 1].x
    #             y2 = path[i + 1].y
    #
    #             x3 = path[j].x
    #             y3 = path[j].y
    #             x4 = path[j + 1].x
    #             y4 = path[j + 1].y
    #
    #             # Check for intersection (more robust version)
    #             denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    #
    #             if denominator != 0:  # Lines are not parallel
    #                 ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    #                 ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator
    #
    #                 if 0 <= ua <= 1 and 0 <= ub <= 1:  # Intersection found
    #                     crossover_count += 1
    #
    #     return crossover_count

    def count_crowdings(self):

        start = time.perf_counter_ns() // 1000

        #Draw on path
        drawing_on_white = np.zeros((self.dims[0], self.dims[1]), dtype=np.uint8)
        for i in range(len(self.tour_path) - 1):  # Iterate through nodes, drawing lines between them
            start_point = (self.tour_path[i].x, self.tour_path[i].y)  # Cast to int for pixel indexing
            end_point = (self.tour_path[i + 1].x, self.tour_path[i + 1].y)
            cv2.line(drawing_on_white, start_point, end_point, (255,255,255), 1)
        binarized = drawing_on_white.astype(bool)
        binarized_inverse = ~binarized
        end = time.perf_counter_ns() // 1000
        # print("crossovers, drawing: "+ str(float(end-start)/1000.0))

        start = time.perf_counter_ns() // 1000
        #Assess continuous blocks of white space, small size indicats many crossovers
        # Label connected components (blocks)
        _, broken_up = label(binarized_inverse)
        end = time.perf_counter_ns() // 1000

        #Assess how many parallel lines
        parallels_kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded_image = cv2.erode(drawing_on_white, parallels_kernal, iterations=1)
        parallel_pixels = np.count_nonzero(eroded_image)

        self.stats.crowding = broken_up + parallel_pixels

        #The more small blocks, the more broken up the image
        # print("crossovers, labelling: "+ str(float(end-start)/1000.0))


