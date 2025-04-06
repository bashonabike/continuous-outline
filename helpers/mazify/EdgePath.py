import numpy as np
from scipy.ndimage import convolve1d

# import helpers.mazify.temp_options as options
import helpers.mazify.EdgeNode as EdgeNode
import helpers.mazify.MazeSections as sections
from helpers.Enums import NodeType

class EdgePath:
    def __init__(self, parent_inkex, options, path_num, path_raw, maze_sections: sections.MazeSections, is_outer=False,
                 max_inner_contour_len=0, from_db=False, is_closed=None, custom_weight=None,
                 num_nodes=None):
        self.options = options
        self.parent_inkex = parent_inkex
        if not from_db:
            self.path, self.num = [], path_num
            self.from_db_num_nodes = 0
            self.outer = is_outer
            self.closed = True #assert closed
            self.section_tracker, self.section_tracker_red_nd_doubled = [], None
            self.custom_weight = 0
            if self.options.inner_contour_variable_weights and not is_outer and max_inner_contour_len > 0:
                #Set custom weight based on contour length
                self.custom_weight = self.options.dumb_node_optional_weight + ((max_inner_contour_len - len(path_raw))//
                          (max_inner_contour_len//
                           (self.options.dumb_node_optional_max_variable_weight - self.options.dumb_node_optional_weight + 1)))
                self.parse_path(path_raw, maze_sections, is_outer, self.custom_weight)
            else:
                self.parse_path(path_raw, maze_sections, is_outer)
        else:
            self.path, self.num = None, path_num
            self.outer = is_outer
            self.closed = is_closed
            self.section_tracker = None
            self.custom_weight = custom_weight
            self.from_db_num_nodes = num_nodes

    @classmethod
    def from_df(cls, options, path_num, is_outer, is_closed, custom_weight, num_nodes):
        return cls(options, path_num, None,  None, is_outer=is_outer, from_db=True, is_closed=is_closed,
                   custom_weight=custom_weight, num_nodes=num_nodes)

    def set_path_and_trackers(self, path, trackers):
        self.path = path
        self.section_tracker = trackers
        self.section_tracker_red_nd_doubled = \
            np.array([t.section for t in (self.section_tracker + self.section_tracker)])

    def parse_path(self, path, maze_sections: sections.MazeSections, is_outer=False,
       custom_weight=0):
        #Set up nodes
        prev_section, section_tracker_num = None, -1
        prev_tracker, cur_tracker = None, None
        first_graph_node, prev_graph_node, last_graph_node = None, None, None
        first_section = None
        edge_weight = self.options.dumb_node_required_weight if is_outer else self.options.dumb_node_optional_weight
        if custom_weight > 0: edge_weight = custom_weight
        is_focus, section_edge_weight = False, edge_weight
        for i in range(len(path)):
            node = EdgeNode.EdgeNode(path[i][0], path[i][1], self, i, is_outer)

            if i > 0:
                node.set_prev_node(self.path[i-1])
                self.path[i-1].set_next_node(node)
            self.path.append(node)
            cur_section = maze_sections.get_section_from_coords(node.y, node.x)
            cur_section.add_node(node)
            if cur_section is not prev_section:
                section_tracker_num += 1
                #If not focus region, turn off (persist if prev was focus region)
                if cur_section.focus_region or (prev_section is not None and prev_section.focus_region): is_focus = True
                section_edge_weight = self.options.dumb_node_required_weight if is_focus else edge_weight

                cur_tracker = sections.MazeSectionTracker(self.options, cur_section, node, section_tracker_num,
                                                          prev_tracker=prev_tracker)
                if section_tracker_num > 0: self.section_tracker[-1].next_tracker = cur_tracker
                self.section_tracker.append(cur_tracker)
                #NOTE: this one lags behind by one
                if section_tracker_num > 0:
                    self.section_tracker[section_tracker_num - 1].out_node = self.path[i-1]
                    self.section_tracker[section_tracker_num - 1].rev_in_node = self.path[i-1]

                #Set graph nodes in
                cur_graph_node = (cur_section.y_sec, cur_section.x_sec, self.num, section_tracker_num)
                graph_node_type = NodeType.section_tracker_req if cur_section.dumb_req else NodeType.section_tracker_opt
                maze_sections.path_graph.add_node(cur_graph_node, category=graph_node_type)
                if first_graph_node is None:
                    first_graph_node = cur_graph_node
                else:
                    maze_sections.path_graph.add_edge(prev_graph_node, cur_graph_node, weight=section_edge_weight)

                maze_sections.path_graph.add_edge(cur_graph_node, (cur_section.y_sec, cur_section.x_sec),
                                                  weight=self.options.dumb_node_req_jump_weight if is_focus or is_outer
                                                  else self.options.dumb_node_opt_jump_weight)
                if is_outer or cur_section.focus_region:
                    cur_section.dumb_req = True

                    #Set into outer path graph
                    maze_sections.outer_paths_graph.add_node(cur_graph_node)
                    cur_sect_blank_node = (cur_section.y_sec, cur_section.x_sec)
                    if not cur_sect_blank_node in maze_sections.outer_paths_graph:
                        maze_sections.outer_paths_graph.add_node(cur_sect_blank_node)
                    maze_sections.outer_paths_graph.add_edge(cur_sect_blank_node, cur_graph_node, weight=10)
                    if prev_graph_node is not None:
                        maze_sections.outer_paths_graph.add_edge(prev_graph_node, cur_graph_node, weight=1)
                else:
                    cur_section.dumb_opt = True

                prev_tracker = cur_tracker
                prev_section = cur_section
                if first_section is None: first_section = cur_section
                last_graph_node = prev_graph_node = cur_graph_node

            node.set_section(cur_section, cur_tracker)
            cur_tracker.nodes.append(node)

        self.section_tracker[-1].out_node = self.path[-1]
        self.section_tracker[-1].rev_in_node = self.section_tracker[-1].out_node
        self.section_tracker[0].rev_in_node = self.section_tracker[0].out_node

        #Check if contour is closed
        circ_dist = 0
        circ_dist += abs(self.path[0].y - self.path[-1].y)
        circ_dist += abs(self.path[0].x - self.path[-1].x)
        if circ_dist > self.options.max_inner_path_seg_manhatten_length + 1:
            self.closed = False
        else:
            self.section_tracker[-1].next_tracker = self.section_tracker[0]
            self.section_tracker[0].prev_tracker = self.section_tracker[-1]
            self.path[-1].set_next_node(self.path[0])
            self.path[0].set_prev_node(self.path[-1])

            if first_section.focus_region: section_edge_weight = self.options.dumb_node_required_weight
            maze_sections.path_graph.add_edge(last_graph_node, first_graph_node, weight=section_edge_weight)
            maze_sections.outer_paths_graph.add_edge(last_graph_node, first_graph_node, weight=1)

        self.section_tracker_red_nd_doubled = np.array([t.section for t in (self.section_tracker + self.section_tracker)])

    def get_next_node(self, cur_node, edge_rev:bool):
        if edge_rev:
            return self.path[(len(self.path) + self.path.index(cur_node) - 1) % len(self.path)]
        return self.path[(self.path.index(cur_node) + 1) % len(self.path)]


    def gaussian_kernel_1d(self, kernel_size, sigma):
        """
        Generates a 1D normalized Gaussian kernel using NumPy.

        Args:
            kernel_size: The size of the kernel (must be odd).
            sigma: The standard deviation of the Gaussian distribution.

        Returns:
            A 1D NumPy array representing the normalized Gaussian kernel.
        """

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        center = kernel_size // 2
        x = np.arange(kernel_size) - center

        kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))

        # Normalize the kernel
        kernel /= np.sum(kernel)

        return kernel
