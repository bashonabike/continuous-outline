import math
import numpy as np
from scipy.ndimage import convolve1d

import helpers.mazify.temp_options as options
import helpers.mazify.EdgeNode as EdgeNode
import helpers.mazify.MazeSections as sections
from helpers.Enums import NodeType

class EdgePath:
    def __init__(self, path_num, path_raw, maze_sections: sections.MazeSections, is_outer=False):
        self.path, self.num = [], path_num
        self.outer = is_outer
        self.section_tracker, self.section_tracker_red_nd_doubled = [], None
        self.parse_path(path_raw, maze_sections, is_outer)


    def parse_path(self, path, maze_sections: sections.MazeSections, is_outer=False):
        #Determine explicit directions
        dys,dxs = [],[]
        for i in range(len(path)):
            i_next = (i + 1) % len(path)
            dy, dx = path[i_next][0] - path[i][0], path[i_next][1] - path[i][1]
            dys.append(dy)
            dxs.append(dx)
        dy_nd, dx_nd = np.array(dys), np.array(dxs)
        path_fwd_dirs_nd = np.arctan2(dy_nd, dx_nd)
        path_fwd_dirs = path_fwd_dirs_nd.tolist()
        path_rev_dirs_part = (path_fwd_dirs_nd + np.pi) % (2 * np.pi)
        path_rev_dirs = np.append(path_rev_dirs_part[-1], path_rev_dirs_part[:-1]).tolist()
        path_displs = np.sqrt(dy_nd**2 + dx_nd**2).tolist()

        #Set smoothed directions
        kernel = self.gaussian_kernel_1d(options.dir_smoothing_size, options.dir_smoothing_sigma)
        smoothed_dy_nd, smoothed_dx_nd = convolve1d(dy_nd, kernel, mode='wrap'),  convolve1d(dx_nd, kernel, mode='wrap')
        path_fwd_dirs_smoothed_nd = np.arctan2(smoothed_dy_nd, smoothed_dx_nd)
        path_fwd_dirs_smoothed = path_fwd_dirs_smoothed_nd.tolist()
        path_rev_dirs_smoothed_part = (path_fwd_dirs_smoothed_nd + np.pi) % (2 * np.pi)
        path_rev_dirs_smoothed = np.append(path_rev_dirs_smoothed_part[-1],
                                                path_rev_dirs_smoothed_part[:-1]).tolist()

        #Set up nodes
        prev_section, section_tracker_num = None, -1
        prev_tracker, cur_tracker = None, None
        first_graph_node, prev_graph_node, last_graph_node = None, None, None
        edge_weight = options.dumb_node_required_weight if is_outer else options.dumb_node_optional_weight
        for i in range(len(path)):
            node = EdgeNode.EdgeNode(path[i][0], path[i][1], self, path_rev_dirs[i], path_rev_dirs_smoothed[i],
                                     path_fwd_dirs[i], path_fwd_dirs_smoothed[i], path_displs[i], is_outer)

            if i > 0:
                node.set_prev_node(self.path[i-1])
                self.path[i-1].set_next_node(node)
            self.path.append(node)
            cur_section = maze_sections.get_section_from_coords(node.y, node.x)
            cur_section.add_node(node)
            if cur_section is not prev_section:
                section_tracker_num += 1
                cur_tracker = sections.MazeSectionTracker(cur_section, node, section_tracker_num,
                                                          prev_tracker=prev_tracker)
                if section_tracker_num > 0: self.section_tracker[-1].next_tracker = cur_tracker
                self.section_tracker.append(cur_tracker)
                cur_section.paths.add(self)
                if self.outer: cur_section.outer_paths.add(self)
                else: cur_section.inner_paths.add(self)
                #NOTE: this one lags behind by one
                if section_tracker_num > 0: self.section_tracker[section_tracker_num - 1].out_node = self.path[i-1]

                #Set graph nodes in
                cur_graph_node = (cur_section.y_sec, cur_section.x_sec, self.num, section_tracker_num)
                graph_node_type = NodeType.section_tracker_req if cur_section.dumb_req else NodeType.section_tracker_opt
                maze_sections.path_graph.add_node(cur_graph_node, category=graph_node_type)
                if first_graph_node is None:
                    first_graph_node = cur_graph_node
                else:
                    maze_sections.path_graph.add_edge(prev_graph_node, cur_graph_node, weight=edge_weight)

                maze_sections.path_graph.add_edge(cur_graph_node, (cur_section.y_sec, cur_section.x_sec),
                                                      weight=options.dumb_node_blank_weight)

                prev_tracker = cur_tracker
                prev_section = cur_section
                last_graph_node = prev_graph_node = cur_graph_node

            node.set_section(cur_section, cur_tracker)
            cur_tracker.nodes.append(node)

        maze_sections.path_graph.add_edge(last_graph_node, first_graph_node, weight=edge_weight)
        self.section_tracker[-1].out_node = self.path[-1]
        self.section_tracker[-1].next_tracker = self.section_tracker[0]
        self.section_tracker[0].prev_tracker = self.section_tracker[-1]
        self.path[-1].set_next_node(self.path[0])
        self.path[0].set_prev_node(self.path[-1])
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
