import numpy as np
from scipy.signal import convolve2d

import helpers.mazify.temp_options as options

class MazeSections:
    def __init__(self, outer_edge, m, n):
        self.m, self.n = m, n
        self.outer_edge = outer_edge
        self.num_sections = m * n
        self.sections_satisfied = 0
        self.sections_satisfied_pct = 0.0
        self.sections, self.section_indices_list, self.y_grade, self.x_grade = self.count_true_pixels_in_sections(outer_edge, m, n)

    def update_saturation(self):
        self.sections_satisfied += 1
        self.sections_satisfied_pct = self.sections_satisfied / self.num_sections

    def check_saturation(self):
        return self.sections_satisfied_pct > options.saturation_termination


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

                # Convolve with ones to find tightest cluster
                kernel = np.ones((options.cluster_start_point_size, options.cluster_start_point_size), dtype=np.uint8)
                convolved = convolve2d(section.astype(np.uint8), kernel, mode='same')
                max_index = np.argmax(convolved)
                max_clust_y, max_clust_x = np.unravel_index(max_index, section.shape)

                # Count True pixels
                count = np.count_nonzero(section)
                sections[i, j] = MazeSection(self, (y_start, y_end, x_start, x_end), count, i, j,
                                             (max_clust_y, max_clust_x))

        return sections, section_indices_list, section_height, section_width

    def get_section_from_coords(self, y, x):
        return self.sections[min(y // self.y_grade, self.m - 1), min(x // self.x_grade, self.n - 1)]

    def get_section_indices_from_coords(self, y, x):
        return min(y // self.y_grade, self.m - 1), min(x // self.x_grade, self.n - 1)

class MazeSection:
    def __init__(self, parent:MazeSections, bounds, edge_pixels, y_sec, x_sec, cluster_point_rel):
        (self.ymin, self.ymax, self.xmin, self.xmax) = bounds
        self.y_sec, self.x_sec = y_sec, x_sec
        self.edge_pixels = edge_pixels
        self.filled_pixels = 0
        self.saturation = 0.0 if edge_pixels > 0 else 1.0
        self.saturated = False if edge_pixels > 0 else True
        if self.saturated: parent.update_saturation()
        self.attraction = 100.0
        self.cluster_point_abs = (self.ymin + cluster_point_rel[0], self.xmin + cluster_point_rel[1])
        self.nodes = []


    def update_saturation(self, parent:MazeSections, fill_count):
        #TODO: improve this so it doesn't double-count saturation
        self.filled_pixels += fill_count
        self.saturation = float(self.filled_pixels) / self.edge_pixels
        self.attraction = 1.0/(self.saturation + 0.01)
        if not self.saturated and self.saturation >= options.section_saturation_satisfied:
            self.saturated = True
            parent.update_saturation()

    def add_node(self, node):
        self.nodes.append(node)

    def get_nodes_by_edge_number(self, path_number):
        return [node for node in self.nodes if node.path_number == path_number]

    def get_surrounding_nodes_by_edge__number(self, parent:MazeSections, path_number):
        nodes = []
        for y_sec in range(max(0, self.y_sec - 1), min(parent.m, self.y_sec + 2)):
            for x_sec in range(max(0, self.x_sec - 1), min(parent.n, self.x_sec + 2)):
                nodes.extend(parent.sections[y_sec, x_sec].get_nodes_by_edge_number(path_number))

        return nodes
