import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d

import helpers.mazify.temp_options as options
import helpers.mazify.MazeAgentHelpers as helpers
import helpers.mazify.NetworkInputs as inputs
import helpers.mazify.MazeSections as sections
import helpers.mazify.TestGetDirection as getdir
from helpers.Enums import CompassType, CompassDir
from helpers.mazify.EdgePath import EdgePath
from helpers.mazify.EdgeNode import EdgeNode
import helpers.mazify.ShortestPathCoverage as shorty

class MazeAgent:
    def __init__(self, outer_edges, outer_contours, inner_edges, inner_contours,
                 maze_sections: sections.MazeSections):
        self.path, self.path_nd = [], np.array([])
        self.unique_segments, self.unique_segments_list, self.unique_segments_centroid_nd = (
            set([]), [], np.array([]))
        #NOTE: edges are codified numerically to correspond with outer contours
        self.outer_edges, self.outer_contours = outer_edges, outer_contours
        self.inner_edges, self.inner_contours = inner_edges, inner_contours

        self.all_edges_bool, self.all_contours = (np.where(self.outer_edges + self.inner_edges > 0, True, False),
                                             self.outer_contours + self.inner_contours)
        self.all_contours_objects = []
        for i in range(len(self.all_contours)):
            self.all_contours_objects.append(EdgePath(i + 1, self.all_contours[i], maze_sections,
                                             i < len(self.outer_contours)))
        maze_sections.set_dumb_nodes()
        req_dumb_nodes = [n for n in range(1, maze_sections.dumb_opt_node_start)]
        test_path = shorty.find_shortest_path_with_coverage(maze_sections.dumb_nodes_weighted,
                                                            maze_sections.dumb_nodes_req,
                                                            req_dumb_nodes)



        self.dims = (outer_edges.shape[0], outer_edges.shape[1])
        self.maze_sections = maze_sections
        self.helper = helpers.MazeAgentHelpers()
        self.cur_section = None
        self.cur_point, self.cur_node = (0, 0), None
        self.prev_direction = -1
        self.inst_directions = []
        self.edge_rev = False

        self.compass_defs =[
            {"type": CompassType.legality_compass, "instantiate": self.legality_check,
             "persist": False, "scalar": False, "on_edge": False, "off_edge": True,
             "custom_normalizer": None},
            {"type": CompassType.proximity_compass, "instantiate": self.proximity_to_edge,
             "persist": False, "scalar": False, "on_edge": False, "off_edge": True,
             "custom_normalizer": None},
            # {"type": CompassType.intersects_compass, "instantiate": self.check_intersects,
            #  "persist": False, "scalar": False, "on_edge": False, "off_edge": True,
            #              "custom_normalizer": None},
            {"type": CompassType.outer_attraction_compass, "instantiate": self.check_outer_attraction,
             "persist": True, "scalar": False, "on_edge": True, "off_edge": True,
             "custom_normalizer": None},
            # {"type": CompassType.parallels_compass, "instantiate": self.check_parallels,
            #  "persist": False, "scalar": False, "on_edge": False, "off_edge": True,
            #              "custom_normalizer": None},
            {"type": CompassType.deflection_compass, "instantiate": self.check_deflection,
             "persist": False, "scalar": False, "on_edge": False, "off_edge": True,
             "custom_normalizer": None},
            {"type": CompassType.inner_attraction, "instantiate": self.check_inner_attraction,
             "persist": False, "scalar": True, "on_edge": False, "off_edge": True,
             "custom_normalizer": 100},
            {"type": CompassType.edge_magnetism, "instantiate": self.check_edge_magenetism,
             "persist": False, "scalar": True, "on_edge": True, "off_edge": False,
             "custom_normalizer": 100*options.edge_magnetism_look_ahead_sections}
        ]
        self.compasses = {}
        self.compass_normalizer = 0.0

        self.network_inputs = inputs.NetworkInputs()
        self.build_inputs()

    #region Build
    def build_inputs(self):
        for compass_def in self.compass_defs:
            if not compass_def['scalar']:
                for compass_dir in CompassDir:
                    self.network_inputs.add_input(compass_def['type'], compass_dir, compass_def['on_edge'],
                                                  compass_def['off_edge'])
            else:
                self.network_inputs.add_input(compass_def['type'], None, compass_def['on_edge'],
                                                  compass_def['off_edge'])
    #endregion
    #region Run
    def plot_path(self, path_coords, image):
        """
        Plots a path defined by a list of tuple coordinates.

        Args:
            path_coords: A list of tuples, where each tuple represents (x, y) coordinates.
        """

        if not path_coords:
            print("Path is empty.")
            return

        x_coords, y_coords = zip(*path_coords)

        plt.imshow(image)  # Display the image
        plt.plot(x_coords, y_coords, color='red', linewidth=1, marker='o', markersize=1)  # Plot the path

        plt.axis('off')  # Turn off axis labels and ticks
        plt.show(block=True)

    def run_round_dumb(self, im_orig_path):
        self.cur_node, self.cur_point, self.cur_section = self.find_start_node()
        self.set_direction_vectors()
        self.set_compasses(on_edge=True)
        self.prev_direction, _ = getdir.get_direction(self.network_inputs, on_edge=True)
        image = Image.open(im_orig_path)

        while not self.maze_sections.check_saturation():
            self.walk_edge_until_exit_section()
            self.cur_node = None
            while self.cur_node is None:
                self.set_direction_vectors()
                self.set_compasses(off_edge=True)
                direction, _ = getdir.get_direction(self.network_inputs, off_edge=True)
                #TODO: Configure with staying power factored in
                self.cur_node = self.check_intersect_edge_update_point(direction)
            self.plot_path(self.path, image) #TEMPPP

    def set_compass(self, compass_type):
        instantiate = None
        for compass_def in self.compass_defs:
            if compass_def['type'] == compass_type:
                instantiate = compass_def['instantiate']
                break
        self.compasses[compass_type] = instantiate()


    def set_compasses(self, on_edge=False, off_edge=False):
        #Set compasses as needed
        for compass_def in self.compass_defs:
            if on_edge and not compass_def['on_edge']: continue
            if off_edge and not compass_def['off_edge']: continue
            start= time.perf_counter_ns()
            if self.compasses.get(compass_def['type']) is None or not compass_def['persist']:
                self.compasses[compass_def['type']] = compass_def['instantiate']()
                if compass_def['custom_normalizer'] is not None:
                    self.compasses[compass_def['type']]/= compass_def['custom_normalizer']

            end= time.perf_counter_ns()
            print(f"{compass_def['type']}: {(end-start)/1000000} ms")

        #Find normalizer
        compass_points_flat = []
        for compass in self.compasses.values():
            if isinstance(compass, dict):
                compass_points_flat.extend(compass.values())
            else:
                #Scalars must be custom normalized
                continue
        self.compass_normalizer = max(compass_points_flat)

        #Set inputs
        for compass_type, compass in self.compasses.items():
            if isinstance(compass, dict):
                for compass_dir, compass_val in compass.items():
                    cur_input = self.network_inputs.find_input(compass_type, compass_dir)
                    cur_input.set_value(compass_val/self.compass_normalizer)
            else:
                #Scalars must be custom normalized
                cur_input = self.network_inputs.find_input(compass_type, None)
                cur_input.set_value(compass)

    def find_start_node(self):
        # Convolve with ones to find tightest cluster
        kernel = np.ones((options.cluster_start_point_size, options.cluster_start_point_size), dtype=np.uint8)
        convolved = convolve2d(self.all_edges_bool.astype(np.uint8), kernel, mode='same')
        max_index = np.argmax(convolved)
        cluster_point = np.unravel_index(max_index, self.all_edges_bool.shape)
        nearest_outer_point = self.find_closest_outer_edge_point(cluster_point)
        start_node = self.find_closest_node_to_edge_point(nearest_outer_point)
        return start_node, start_node.point, start_node.section
    #endregion
    #region Compassing
    def legality_check(self):
        legal_compass = self.helper.legality_compass(self.inst_directions)
        return legal_compass
    def proximity_to_edge(self):
        #Adding 10E-3 to avoid floating point errors
        proximities = self.helper.process_points_in_quadrant_boxes_to_weighted_centroids(self.cur_point,
                                                                                         self.all_edges_bool,
                                                                                         options.proximity_search_radius)
        proximities_compass = self.helper.compute_compass_from_quadrant_vectors(proximities)
        return proximities_compass


    def check_intersects(self):
        intersects_compass = self.helper.check_intersects_by_direction_compass(self.inst_directions,
                                                                               self.unique_segments_list,
                                                                               self.unique_segments_centroid_nd)
        return intersects_compass

    def check_parallels(self):
        parallels_compass = self.helper.compute_parallels_compass(self.cur_point, self.inst_directions,
                                                                  self.all_edges_bool)
        return parallels_compass

    def check_deflection(self):
        deflection_compass = self.helper.deflection_compass(self.inst_directions, self.prev_direction)
        return deflection_compass

    def check_outer_attraction(self):
        outer_attraction_compass = self.helper.outer_sections_attraction_compass(self.maze_sections, self.cur_section)
        return outer_attraction_compass

    def check_inner_attraction(self):
        inner_attraction_scalar = self.helper.inner_section_attraction_scalar(self.cur_section)
        return inner_attraction_scalar

    def check_edge_magenetism(self):
        edge_magnetism_scalar = self.helper.edge_magnetism_scalar(self.cur_node, self.edge_rev, self.maze_sections)
        return edge_magnetism_scalar

    #endregion
    #region Walkies
    def check_segment_hit_edge(self, start_point, end_point):
        #Walk path one pixel at a time
        dy, dx = end_point[0] - start_point[0], end_point[1] - start_point[1]
        dydx_mag = abs(dy/dx) if dx > 0 else 99999.0
        y_disp, x_disp = 0, 0
        y_dir, x_dir = 1 if dy >= 0 else -1, 1 if dx >= 0 else -1
        while not (abs(y_disp) >= abs(dy) and abs(x_disp) >= abs(dx)):
            try_x = abs(x_disp)*dydx_mag < abs(y_disp)
            if try_x and abs(x_disp) < abs(dx):
                x_disp += x_dir
            elif abs(y_disp) < abs(dy):
                y_disp += y_dir
            elif abs(x_disp) < abs(dx):
                x_disp += x_dir

            #Check if hit edge
            cur_px = (start_point[0] + y_disp, start_point[1] + x_disp)
            if self.all_edges_bool[cur_px]:
                return cur_px

        return None

    def find_closest_node_to_edge_point(self, edge_point):
        #Test outer first, then inner
        edge_num = self.outer_edges[edge_point]

        if edge_num == 0:
            edge_num = self.inner_edges[edge_point]

        if edge_num == 0:
            #Phantom edge weird stuff but not worth breaking
            return None

        #Retrieve specified edge path and find closest node
        edge_path = self.get_edge_path_by_number(edge_num)
        point_section = self.maze_sections.get_section_from_coords(edge_point[0], edge_point[1])
        nodes = point_section.get_nodes_by_edge_number(edge_num)
        if len(nodes) == 0:
            #Expand search to surrounding sections
            nodes = point_section.get_surrounding_nodes_by_edge__number(self.maze_sections, edge_num)
        if len(nodes) == 0:
            #Phantom, ignore
            return None

        nodes_coords = [(node.y, node.x) for node in nodes]
        nodes_coords_nd = np.array(nodes_coords)
        edge_point_nd = np.array(edge_point)
        diff = nodes_coords_nd - edge_point_nd
        squared_dist = np.sum(diff**2, axis=1)
        closest_node = nodes[np.argmin(squared_dist)]
        return closest_node

    def find_closest_outer_edge_point(self, point):
        """
        Finds the closest non-zero point in a 2D NumPy ndarray to a given point.

        Args:
            point: Tuple (x, y) representing the target point.
            array_2d: 2D NumPy ndarray.

        Returns:
            Tuple (x, y) representing the closest non-zero point, or None if no non-zero points exist.
        """

        point = np.array(point)
        nonzero_indices = np.nonzero(self.outer_edges)
        nonzero_points = np.column_stack(nonzero_indices)

        if len(nonzero_points) == 0:
            return None  # No non-zero points found

        distances = np.linalg.norm(nonzero_points - point, axis=1)
        closest_index = np.argmin(distances)
        closest_point = tuple(nonzero_points[closest_index])

        return closest_point



    #endregion
    #region Getters
    def get_edge_path_by_number(self, path_num):
        return self.all_contours_objects[path_num - 1]
    #endregion
    #region Setters
    def walk_edge_until_exit_section(self):
        #Find closest direction to compass pull
        if (abs(self.prev_direction - self.cur_node.fwd_dir_smoothed)
            < abs(self.prev_direction - self.cur_node.rev_dir_smoothed)):
                self.edge_rev = True
        else: self.edge_rev = False
        terminate = False

        num_outer_nodes_x_in_section = 0
        while not terminate:
            #Set node into path
            self.update_point(self.cur_node.point, self.cur_node.rev_dir if self.edge_rev else self.cur_node.fwd_dir)
            if self.cur_node.section is not self.cur_section:
                #Set nodes into old section
                if num_outer_nodes_x_in_section > 0:
                    self.cur_section.update_saturation(self.maze_sections, num_outer_nodes_x_in_section)
                num_outer_nodes_x_in_section = 0

                #Update section
                self.cur_section = self.cur_node.section
                self.set_compasses(on_edge=True)

                #Check if need to exit edge
                ideal_direction, staying_power = getdir.get_direction(self.network_inputs, on_edge=True)
                if staying_power < options.edge_magnetism_cutoff:
                    actual_direction = self.cur_node.rev_dir_smoothed if self.edge_rev else self.cur_node.fwd_dir_smoothed
                    if abs((2 * math.pi) +ideal_direction - actual_direction)%(2 * math.pi) > options.need_to_steer_off_edge:
                        terminate = True
                        self.cur_section.update_saturation(self.maze_sections, 1)
                        self.prev_direction = actual_direction
                        break

            if self.cur_node.path.outer: num_outer_nodes_x_in_section += 1

            #Get next node
            self.cur_node = self.cur_node.path.get_next_node(self.cur_node, self.edge_rev)


    def check_intersect_edge_update_point(self, direction):
        #Get prospective new point and check if intersects with edge
        # total_count, (min_y, max_y, min_x, max_x), sub_counts, new_point = (
        #     self.helper.single_dir_parallels(self.cur_point, self.outer_edges, direction, self.maze_sections))
        new_section, new_point = self.helper.get_next_point(self.cur_point, direction, self.outer_edges,
                                                            self.maze_sections)
        nearest_edge_point = self.check_segment_hit_edge(self.cur_point, new_point)
        if nearest_edge_point is not None:
            #Look for node, if exists
            nearest_edge_node = self.find_closest_node_to_edge_point(nearest_edge_point)
            if nearest_edge_node is not None:
                return nearest_edge_node

        #If not hit edge, continue with forging new path
        if new_section is not self.cur_section:
            self.cur_section = new_section
            self.set_compass(CompassType.outer_attraction_compass)



            # for sub_count in sub_counts:
        #     cur_section = self.maze_sections.sections[(sub_count['y_sec'], sub_count['x_sec'])]
            # if 0 <= sub_count['y_sec'] < self.maze_sections.m and 0 <= sub_count['x_sec'] < self.maze_sections.n and not \
            #     cur_section.saturated:
            #     cur_section.update_saturation(self.maze_sections)

        #Update section if needed
        # if len(sub_counts) > 0:
        #     self.cur_section = self.helper.retrieve_new_section(new_point, self.maze_sections)
        #     self.set_compass(CompassType.outer_attraction_compass)

        #Update point
        self.update_point(new_point, direction)
        return None

    def update_point(self, new_point, direction):
        self.cur_point = new_point
        self.path.append(self.cur_point)
        if len(self.path) >= 2:
            self.path_nd = np.vstack((self.path_nd, np.array(self.cur_point)))
        else:
            self.path_nd = np.array(self.cur_point)

        if len(self.path) >= 2:
            self.unique_segments_list, self.unique_segments_centroid_nd =\
                self.add_sort_segment_to_set(self.path[-1], self.path[-2], self.unique_segments,
                                             self.unique_segments_list, self.unique_segments_centroid_nd)
        self.prev_direction = direction

    def set_direction_vectors(self):
        self.inst_directions = self.helper.parse_direction_vector_starters(self.cur_point, self.dims)

    def add_sort_segment_to_set(self, point1: tuple, point2: tuple, segments: set, segments_list: list,
                                centroids_nd:np.array):
        old_seg_size = len(segments)
        seg_to_add = None
        if point1[0] < point2[0]:
            seg_to_add = (point1, point2)
        elif point2[0] < point1[0]:
            seg_to_add = (point2, point1)
        elif point1[1] < point2[1]:
            seg_to_add = (point1, point2)
        else:
            seg_to_add = (point2, point1)
        segments.add(seg_to_add)

        #Check if actually added
        if len(segments) > old_seg_size:
            segments_list.append(seg_to_add)
            centroid = ((seg_to_add[0][0] + seg_to_add[1][0]) // 2, (seg_to_add[0][1] + seg_to_add[1][1]) // 2)
            if centroids_nd.size > 0:
                centroids_nd = np.vstack((centroids_nd, np.array(centroid)))
            else: centroids_nd = np.array(centroid)
        return segments_list, centroids_nd
    #endregion








