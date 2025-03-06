import numpy as np
from scipy.signal import convolve2d

import helpers.mazify.temp_options as options
from helpers.mazify.EdgeNode import EdgeNode
import copy as cp

class MazeSections:
    def __init__(self, outer_edge, m, n):
        self.m, self.n = m, n
        self.outer_edge = outer_edge
        self.num_sections = m * n
        self.sections_satisfied = 0
        self.sections_satisfied_pct = 0.0
        self.sections, self.section_indices_list, self.y_grade, self.x_grade = self.count_true_pixels_in_sections(outer_edge, m, n)
        # self.dumb_nodes, self.dumb_nodes_req, self.dumb_nodes_opt, self.dumb_opt_node_start = (
        #     np.zeros((m, n), dtype=list), np.zeros((m, n), dtype=list), np.zeros((m, n), dtype=list), -1)
        self.dumb_opt_node_start = -1
        self.dumb_nodes_weighted = np.zeros((m, n), dtype=np.uint8)
        self.dumb_nodes_req = np.zeros((m, n), dtype=np.uint8)
        # self.dumb_nodes.fill([])
        # self.dumb_nodes_req.fill([])
        # self.dumb_nodes_opt.fill([])
        self.dumb_nodes_weighted.fill(options.dumb_node_blank_weight)

    def update_saturation(self):
        self.sections_satisfied += 1
        self.sections_satisfied_pct = self.sections_satisfied / self.num_sections

    def check_saturation(self):
        return self.sections_satisfied_pct > options.saturation_termination

    def set_dumb_nodes(self):

        req_count = 1  # Start counting required elements from 1

        # Process required elements first
        for i in range(self.n):
            for j in range(self.m):
                if self.sections[i, j].dumb_req:
                    # self.dumb_nodes[i][j].append(req_count)
                    # self.dumb_nodes_req[i][j].append(req_count)
                    self.dumb_nodes_weighted[i][j] = options.dumb_node_required_weight
                    if self.dumb_nodes_req[i][j] == 0:
                        self.dumb_nodes_req[i][j] = req_count
                        req_count += 1

        self.dumb_opt_node_start = opt_count = req_count  # Start optional elements from where required left off

        # Process optional elements
        for i in range(self.n):
            for j in range(self.m):
                if self.sections[i, j].dumb_opt:
                    # self.dumb_nodes[i][j].append(opt_count)
                    # self.dumb_nodes_opt[i][j].append(opt_count)
                    if self.dumb_nodes_weighted[i][j] == options.dumb_node_blank_weight:
                        self.dumb_nodes_weighted[i][j] = options.dumb_node_optional_weight
                    elif self.dumb_nodes_weighted[i][j] > options.dumb_node_required_weight + 1:
                        #Reduce impedance a bit if multiple optional paths here
                        self.dumb_nodes_weighted[i][j] -= 1
                    opt_count += 1


    def count_true_pixels_in_sections(self, boolean_image, m, n):
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
                sections[i, j] = MazeSection(self, (y_start, y_end, x_start, x_end), count, i, j)

        return sections, section_indices_list, section_height, section_width

    def get_section_from_coords(self, y, x):
        return self.sections[min(y // self.y_grade, self.m - 1), min(x // self.x_grade, self.n - 1)]

    def get_section_indices_from_coords(self, y, x):
        return min(y // self.y_grade, self.m - 1), min(x // self.x_grade, self.n - 1)

class MazeSection:
    def __init__(self, parent:MazeSections, bounds, edge_pixels, y_sec, x_sec):
        (self.ymin, self.ymax, self.xmin, self.xmax) = bounds
        self.y_sec, self.x_sec = y_sec, x_sec
        self.edge_pixels = edge_pixels
        self.attraction = 100.0
        self.nodes, self.outer_nodes = [], []

        self.filled_nodes = 0
        self.saturation = 0.0
        self.saturated = False
        self.attraction = 100.0

        self.dumb_req = False
        self.dumb_opt = False


    def setup_saturation(self, parent:MazeSections):
        #Only do this AFTER nodes are filled
        self.filled_nodes= 0
        self.saturation = 0.0 if len(self.outer_nodes) > 0 else 1.0
        self.saturated = False if len(self.outer_nodes) > 0 else True
        self.attraction = 100.0 if len(self.outer_nodes) > 0 else 0
        if self.saturated: parent.update_saturation()

    def update_saturation(self, parent:MazeSections, num_nodes):
        if len(self.outer_nodes) == 0: return
        self.filled_nodes += num_nodes
        self.saturation = float(self.filled_nodes) / len(self.outer_nodes)
        self.attraction = 1.0/(self.saturation + 0.01)
        if not self.saturated and self.saturation >= options.section_saturation_satisfied:
            self.saturated = True
            parent.update_saturation()

    def add_node(self, node: EdgeNode, dumb_req=False):
        self.nodes.append(node)
        if node.outer: self.outer_nodes.append(node)
        if dumb_req:
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
