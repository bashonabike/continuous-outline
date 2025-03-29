import numpy as np
# from scipy.signal import convolve2d
import networkx as nx
# import copy as cp

# import helpers.mazify.temp_options as options
from helpers.mazify.EdgeNode import EdgeNode
from helpers.Enums import NodeType

class MazeSections:
    def __init__(self, options, outer_edge, m, n, req_details_mask, from_db=False, focus_region_sections=None,
                 sections=None, y_grade=None, x_grade=None, img_height=None, img_width=None, path_graph=None):
        self.m, self.n = m, n
        self.options = options
        if not from_db:
            self.focus_region_sections = []
            self.img_height, self.img_width = outer_edge.shape
            self.sections, _, self.y_grade, self.x_grade = (
                self.count_true_pixels_in_sections(outer_edge, m, n, req_details_mask))
            self.grid_lines =self.create_grid_paths(self.x_grade, self.y_grade, options.maze_sections_across)
            self.path_graph = nx.Graph()
            self.set_section_blank_overs_in_graph()
        else:
            self.m, self.n = m, n
            self.img_height, self.img_width = img_height, img_width
            self.focus_region_sections = focus_region_sections
            self.sections = sections
            self.y_grade, self.x_grade = y_grade, x_grade
            self.path_graph = path_graph

    @classmethod
    def from_df(cls, options, m, n, focus_region_sections:list, sections:np.ndarray,
                 y_grade, x_grade, img_height, img_width, path_graph: nx.Graph):
        return cls(options, None, m, n, None, from_db=True, focus_region_sections=focus_region_sections,
                   sections=sections, y_grade=y_grade, x_grade=x_grade, img_height=img_height, img_width=img_width,
                   path_graph=path_graph)

    def create_grid_paths(self, x_grade, y_grade, total_intervals):
        """
        Creates a series of grid lines stored as paths.

        Args:
            x_grade: The horizontal spacing between vertical grid lines.
            y_grade: The vertical spacing between horizontal grid lines.
            total_intervals: The total number of intervals in both x and y directions.

        Returns:
            A list of NumPy arrays, where each array represents a grid line path.
        """

        grid_paths = []

        # Create vertical grid lines
        for i in range(total_intervals + 1):
            x = i * x_grade
            vertical_line = [(x, 0), (x, total_intervals * y_grade)]
            grid_paths.append(vertical_line)

        # Create horizontal grid lines
        for i in range(total_intervals + 1):
            y = i * y_grade
            horizontal_line = [(0, y), (total_intervals * x_grade, y)]
            grid_paths.append(horizontal_line)

        return grid_paths

    def set_section_blank_overs_in_graph(self):
        #Set sections as nodes into graph
        for i in range(self.m):
            for j in range(self.n):
                self.path_graph.add_node((i, j))

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
                            self.path_graph.add_edge(current_node, neighbor_node,
                                                     weight=self.options.dumb_node_blank_weight)

    def set_direct_jump_close_nodes(self, parent_inkex):
        def manhatten_dist_nodes(node1, node2):
            return abs(node1.point[0] - node2.point[0]) + abs(node1.point[1] - node2.point[1])

        def check_likely_cross(track_from, track_to):
            #If straddles in either dimension, likely intersection
            for dim in range(2):
                if (min(track_from.in_node.point[dim], track_from.out_node.point[dim]) <
                    min(track_to.in_node.point[dim], track_to.out_node.point[dim]) <=
                    max(track_to.in_node.point[dim], track_to.out_node.point[dim]) <
                    max(track_from.in_node.point[dim], track_from.out_node.point[dim])):
                    return True

            return False

        def check_if_tracks_parallel_or_crisscross(parent_inkex, track_from, track_to):
            edge_node_from_in, edge_node_to_in = track_from.in_node, track_to.in_node
            edge_node_from_out, edge_node_to_out = track_from.out_node, track_to.out_node
            front_to_front_dist = (manhatten_dist_nodes(edge_node_from_in, edge_node_to_in) +
                manhatten_dist_nodes(edge_node_from_out, edge_node_to_out))
            if front_to_front_dist <= (self.y_grade + self.x_grade)//5:
                #Parallel
                return True
            else:
                front_to_back_dist = (manhatten_dist_nodes(edge_node_from_in, edge_node_to_out) +
                    manhatten_dist_nodes(edge_node_from_out, edge_node_to_in))
                if front_to_back_dist <= (self.y_grade + self.x_grade)//5:
                    return True
                elif check_likely_cross(track_from, track_to):
                    return True
                elif check_likely_cross(track_to, track_from):
                    return True

            return False

        for i in range(self.m):
            for j in range(self.n):
                cur_section = self.sections[i, j]
                for track_from in cur_section.section_trackers:
                    for track_to in cur_section.section_trackers:
                        track_from_node = (i, j, track_from.path_num, track_from.tracker_num)
                        track_to_node = (i, j, track_to.path_num, track_to.tracker_num)
                        if track_from != track_to and not self.path_graph.has_edge(track_from_node, track_to_node) and \
                            check_if_tracks_parallel_or_crisscross(parent_inkex, track_from, track_to):
                            self.path_graph.add_edge(track_from_node, track_to_node,
                                                     weight=0)
                            # parent_inkex.msg(f"Jump node {track_from_node} to {track_to_node}")


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

                sections[i, j] = MazeSection(self.options, (y_start, y_end, x_start, x_end), count, i, j,
                                             len(focus_region_nums) > 0, focus_region_nums)
                for k in focus_region_nums: self.focus_region_sections[k].append(sections[i, j])

        return sections, section_indices_list, section_height, section_width

    def get_section_from_coords(self, y, x):
        return self.sections[min(y // self.y_grade, self.m - 1), min(x // self.x_grade, self.n - 1)]

    def get_section_indices_from_coords(self, y, x):
        return min(y // self.y_grade, self.m - 1), min(x // self.x_grade, self.n - 1)

class MazeSection:
    def __init__(self, options, bounds, edge_pixels, y_sec, x_sec, focus_region, focus_region_nums=None, from_df=False,
                 dumb_req=False, dumb_opt=False):
        self.options = options
        (self.ymin, self.ymax, self.xmin, self.xmax) = bounds
        self.y_sec, self.x_sec = y_sec, x_sec
        self.coords_sec = (y_sec, x_sec)
        self.edge_pixels = edge_pixels
        self.nodes, self.outer_nodes = [], []

        self.focus_region = focus_region
        self.focus_region_nums = focus_region_nums
        self.dumb_req = dumb_req
        self.dumb_opt = dumb_opt

        self.section_trackers = []

    @classmethod
    def from_df(cls, options, y_start, y_end, x_start, x_end, num_edge_pixels, y_sec, x_sec,
                is_focus_region, focus_region_nums,
                dumb_req, dumb_opt):
        if len(focus_region_nums) > 0:
            focus_region_nums = [int(n) for n in focus_region_nums.split(",")]
        else:
            focus_region_nums = []
        new_section = cls(options, (y_start, y_end, x_start, x_end), num_edge_pixels, y_sec, x_sec, is_focus_region,
                          focus_region_nums=focus_region_nums, from_df=True, dumb_req=dumb_req, dumb_opt=dumb_opt)
        return new_section


    def bulk_add_nodes_from_df(self, nodes:list[EdgeNode], outer_nodes:list[EdgeNode]):
        self.nodes = nodes
        self.outer_nodes = outer_nodes

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
    def __init__(self, options, section:MazeSection, in_node:EdgeNode=None, tracker_num:int=None,
                 prev_tracker=None, next_tracker=None, out_node:EdgeNode=None, from_db=False, path_num=None,
                 from_db_num_nodes=None):
        self.options = options
        self.section = section
        self.tracker_num = tracker_num
        self.nodes = []
        if not from_db:
            self.in_node = in_node
            self.out_node = out_node
            self.path, self.path_num = in_node.path, in_node.path.num
            self.rev_in_node, self.rev_out_node = out_node, in_node
            self.prev_tracker = prev_tracker
            self.next_tracker = next_tracker
            self.from_db_num_nodes = 0
            section.section_trackers.append(self)
        else:
            self.path_num = path_num
            self.in_node = None
            self.out_node = None
            self.path = None
            self.rev_in_node, self.rev_out_node = None, None
            self.prev_tracker = None
            self.next_tracker = None
            self.from_db_num_nodes = from_db_num_nodes

    @classmethod
    def from_df(cls, options, section:MazeSection, tracker_num:int, path_num:int, num_nodes):
        return cls(options, section, from_db=True, tracker_num=tracker_num, path_num=path_num,
                   from_db_num_nodes=num_nodes)

    def set_nodes_and_neighbours(self, nodes:list, prev_tracker, next_tracker):
        self.nodes = nodes
        self.in_node = nodes[0]
        self.out_node = nodes[-1]
        self.path = self.in_node.path
        self.rev_in_node, self.rev_out_node = nodes[0], nodes[-1]
        self.prev_tracker, self.next_tracker = prev_tracker, next_tracker

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
