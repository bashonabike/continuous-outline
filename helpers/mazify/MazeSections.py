import numpy as np
# from scipy.signal import convolve2d
import networkx as nx
# import copy as cp

import helpers.mazify.temp_options as options
from helpers.mazify.EdgeNode import EdgeNode
from helpers.Enums import NodeType

class MazeSections:
    def __init__(self, outer_edge, m, n, req_details_mask):
        self.m, self.n = m, n
        self.focus_region_sections = []
        self.dumb_nodes_req = np.zeros((m, n), dtype=np.uint8)
        self.sections, self.section_indices_list, self.y_grade, self.x_grade = (
            self.count_true_pixels_in_sections(outer_edge, m, n, req_details_mask))

        self.path_graph = nx.Graph()
        # self.set_section_blank_overs_in_graph()


    # def set_section_blank_overs_in_graph(self):
    #     #Set sections as nodes into graph
    #     for i in range(self.m):
    #         for j in range(self.n):
    #             self.path_graph.add_node((i, j))

        #Set jumps as edges into graph
        for i in range(self.m):
            for j in range(self.n):
                current_node = (i, j)
                neighbors = [
                    (i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                    (i, j - 1), (i, j + 1),
                    (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)
                ]

                for ni, nj in neighbors:
                    if 0 <= ni < self.m and 0 <= nj < self.n:
                        neighbor_node = (ni, nj)
                        # Add edge only if it doesn't already exist (undirected)
                        if not self.path_graph.has_edge(current_node, neighbor_node):
                            self.path_graph.add_edge(current_node, neighbor_node, weight=options.dumb_node_blank_weight)

    def set_section_node_cats(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.sections[i, j].dumb_req:
                    self.path_graph.nodes[(i, j)]['category'] = NodeType.section_req
                elif self.sections[i, j].dumb_opt:
                    self.path_graph.nodes[(i, j)]['category'] = NodeType.section_opt
                else:
                    self.path_graph.nodes[(i, j)]['category'] = NodeType.section_blank

        for i in range(len(self.focus_region_sections)):
            self.focus_region_sections[i] = [s for s in self.focus_region_sections[i] if s is not None
                                             and (s.dumb_req or s.dumb_opt)]
    def count_true_pixels_in_sections(self, boolean_image, m, n, req_details_masks:list[np.ndarray]):
        """
        Breaks a boolean image into m x n rectangular sections and counts the number of
        True pixels in each section.

        Args:
            boolean_image (numpy.ndarray): The boolean image (True/False or 1/0).
            m (int): The number of rows of sections.
            n (int): The number of columns of sections.

        Returns:
            numpy.ndarray: A 2D array where each element represents the count of True
                           pixels in the corresponding section.
        """

        height, width = boolean_image.shape
        section_height = height // m
        section_width = width // n

        # Handle cases where the image cannot be divided evenly
        remainder_height = height % m
        remainder_width = width % n

        sections = np.zeros((m, n), dtype=MazeSection)
        section_indices_list = []

        self.focus_region_sections = [[] for _ in range(len(req_details_masks))]

        for i in range(m):
            for j in range(n):
                # Calculate section boundaries, handling remainders
                y_start = i * section_height
                y_end = (i + 1) * section_height + (1 if i == m - 1 and remainder_height > 0 else 0)
                x_start = j * section_width
                x_end = (j + 1) * section_width + (1 if j == n - 1 and remainder_width > 0 else 0)

                # Extract the section
                section = boolean_image[y_start:y_end, x_start:x_end]
                section_indices_list.append((i, j))

                # Count True pixels
                count = np.count_nonzero(section)

                #Check if in details req
                focus_region_nums = []
                for k in range(len(req_details_masks)):
                    req_section = req_details_masks[k][y_start:y_end, x_start:x_end]
                    if np.any(req_section):
                        focus_region_nums.append(k)

                sections[i, j] = MazeSection(self, (y_start, y_end, x_start, x_end), count, i, j,
                                             len(focus_region_nums) > 0, focus_region_nums)
                for k in focus_region_nums: self.focus_region_sections[k].append(sections[i, j])
                if len(focus_region_nums) > 0: self.dumb_nodes_req[i, j] = 1

        return sections, section_indices_list, section_height, section_width

    def get_section_from_coords(self, y, x):
        return self.sections[min(y // self.y_grade, self.m - 1), min(x // self.x_grade, self.n - 1)]

    def get_section_indices_from_coords(self, y, x):
        return min(y // self.y_grade, self.m - 1), min(x // self.x_grade, self.n - 1)

class MazeSection:
    def __init__(self, parent:MazeSections, bounds, edge_pixels, y_sec, x_sec, focus_region, focus_region_nums=None):
        (self.ymin, self.ymax, self.xmin, self.xmax) = bounds
        self.y_sec, self.x_sec = y_sec, x_sec
        self.coords_sec = (y_sec, x_sec)
        self.edge_pixels = edge_pixels
        self.nodes, self.outer_nodes = [], []

        self.focus_region = focus_region
        self.focus_region_nums = focus_region_nums
        self.dumb_req = False
        self.dumb_opt = False

    def add_node(self, node: EdgeNode):
        self.nodes.append(node)
        if node.outer:
            self.outer_nodes.append(node)
            self.dumb_req = True
        elif self.focus_region:
            self.dumb_req = True
        else:
            self.dumb_opt = True


    def get_nodes_by_edge_number(self, path_number):
        return [node for node in self.nodes if node.path_num == path_number]

    def get_surrounding_nodes_by_edge__number(self, parent:MazeSections, path_number):
        nodes = []
        for y_sec in range(max(0, self.y_sec - 1), min(parent.m, self.y_sec + 2)):
            for x_sec in range(max(0, self.x_sec - 1), min(parent.n, self.x_sec + 2)):
                nodes.extend(parent.sections[y_sec, x_sec].get_nodes_by_edge_number(path_number))

        return nodes

class MazeSectionTracker:
    def __init__(self, section:MazeSection, in_node:EdgeNode, tracker_num:int,
                 prev_tracker=None, next_tracker=None, out_node:EdgeNode=None):
        self.section = section
        self.nodes = []
        self.in_node = in_node
        self.out_node = out_node
        self.path, self.path_num = in_node.path, in_node.path.num
        self.rev_in_node, self.rev_out_node = out_node, in_node
        self.tracker_num = tracker_num
        self.prev_tracker = prev_tracker
        self.next_tracker = next_tracker

    def __hash__(self):
        return hash((self.section.y_sec, self.section.x_sec, self.path_num, self.tracker_num))

    def __eq__(self, other):
        if not isinstance(other, MazeSectionTracker):
            return False
        return (self.section.y_sec == other.section.y_sec and
                self.section.x_sec == other.section.x_sec and
                self.path_num == other.path_num and
                self.tracker_num == other.tracker_num)

    def __lt__(self, other):
        if not isinstance(other, MazeSectionTracker):
            return NotImplemented
        if self.section.y_sec < other.section.y_sec:
            return True
        elif self.section.y_sec > other.section.y_sec:
            return False
        if self.section.x_sec < other.section.x_sec:
            return True
        elif self.section.x_sec > other.section.x_sec:
            return False
        if self.path_num < other.path_num:
            return True
        elif self.path_num > other.path_num:
            return False
        return self.tracker_num < other.tracker_num

    def __gt__(self, other):
        if not isinstance(other, MazeSectionTracker):
            return NotImplemented
        if self.section.y_sec > other.section.y_sec:
            return True
        elif self.section.y_sec < other.section.y_sec:
            return False
        if self.section.x_sec > other.section.x_sec:
            return True
        elif self.section.x_sec < other.section.x_sec:
            return False
        if self.path_num > other.path_num:
            return True
        elif self.path_num < other.path_num:
            return False
        return self.tracker_num > other.tracker_num

    def get_next_tracker(self, reverse=False):
        if reverse:
            return self.prev_tracker
        else:
            return self.next_tracker

    def get_in_node(self, reverse=False):
        if reverse:
            return self.rev_in_node
        else:
            return self.in_node

    def get_out_node(self, reverse=False):
        if reverse:
            return self.rev_out_node
        else:
            return self.out_node
