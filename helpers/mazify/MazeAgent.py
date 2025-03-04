import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

import helpers.mazify.temp_options as options
import helpers.mazify.MazeAgentHelpers as helpers
import helpers.mazify.NetworkInputs as inputs
import helpers.mazify.MazeSections as sections
import helpers.mazify.TestGetDirection as getdir
from helpers.Enums import CompassType, CompassDir
from helpers.mazify.EdgePath import EdgePath

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
            self.all_contours_objects.append(EdgePath(i + 1, self.all_contours[i], maze_sections))

        self.dims = (outer_edges.shape[0], outer_edges.shape[1])
        self.maze_sections = maze_sections
        self.helper = helpers.MazeAgentHelpers()
        self.cur_section = None
        self.cur_point = (0, 0)
        self.prev_direction = -1
        self.inst_directions = []

        self.compass_defs =[
            {"type": CompassType.legality_compass, "instantiate": self.legality_check,
             "persist": False, "scalar": False},
            {"type": CompassType.proximity_compass, "instantiate": self.proximity_to_edge,
             "persist": False, "scalar": False},
            {"type": CompassType.intersects_compass, "instantiate": self.check_intersects,
             "persist": False, "scalar": False},
            {"type": CompassType.outer_attraction_compass, "instantiate": self.check_outer_attraction,
             "persist": True, "scalar": False},
            {"type": CompassType.parallels_compass, "instantiate": self.check_parallels,
             "persist": False, "scalar": False},
            {"type": CompassType.deflection_compass, "instantiate": self.check_deflection,
             "persist": False, "scalar": False},
            {"type": CompassType.inner_attraction, "instantiate": self.check_inner_attraction,
             "persist": False, "scalar": True},
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
                    self.network_inputs.add_input(compass_def['type'], compass_dir)
            else:
                self.network_inputs.add_input(compass_def['type'], None)
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
        self.cur_point, self.cur_section = self.find_start_point()
        self.path.append(self.cur_point)
        self.path_nd = np.array(self.cur_point)
        image = Image.open(im_orig_path)

        while not self.maze_sections.check_saturation():
            self.set_direction_vectors()
            self.set_compasses()
            direction = getdir.get_direction(self.network_inputs)
            self.update_section_saturation_and_point(direction)
            if len(self.path)%100 == 0:
                self.plot_path(self.path, image)

    def set_compass(self, compass_type):
        instantiate = None
        for compass_def in self.compass_defs:
            if compass_def['type'] == compass_type:
                instantiate = compass_def['instantiate']
                break
        self.compasses[compass_type] = instantiate()


    def set_compasses(self):
        #Set compasses as needed
        for compass_def in self.compass_defs:
            start= time.perf_counter_ns()
            if self.compasses.get(compass_def['type']) is None or not compass_def['persist']:
                self.compasses[compass_def['type']] = compass_def['instantiate']()

            end= time.perf_counter_ns()
            print(f"{compass_def['type']}: {(end-start)/1000000} ms")

        #Find normalizer
        compass_points_flat = []
        for compass in self.compasses.values():
            if isinstance(compass, dict):
                compass_points_flat.extend(compass.values())
            else:
                compass_points_flat.append(compass)
        self.compass_normalizer = max(compass_points_flat)

        #Set inputs
        for compass_type, compass in self.compasses.items():
            if isinstance(compass, dict):
                for compass_dir, compass_val in compass.items():
                    cur_input = self.network_inputs.find_input(compass_type, compass_dir)
                    cur_input.set_value(compass_val/self.compass_normalizer)
            else:
                cur_input = self.network_inputs.find_input(compass_type, None)
                cur_input.set_value(compass/self.compass_normalizer)

    def find_start_point(self):
        #TODO: cluster whole maze

        #Find start section
        edge_pixels_array= np.vectorize(lambda x: x.edge_pixels)(self.maze_sections.sections)
        max_edges_index = np.argmax(edge_pixels_array)
        start_maze_section = self.maze_sections.sections.flat[max_edges_index]

        #Start point in section one max clustered
        start_point = start_maze_section.cluster_point_abs
        return start_point, start_maze_section
    #endregion
    #region Compassing
    def legality_check(self):
        legal_compass = self.helper.legality_compass(self.inst_directions)
        return legal_compass
    def proximity_to_edge(self):
        #Adding 10E-3 to avoid floating point errors
        proximities = self.helper.process_points_in_quadrant_boxes_to_weighted_centroids(self.cur_point, self.all_edges,
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
                                                                  self.all_edges)
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





    #endregion
    #region Getters
    def get_edge_path_by_number(self, path_num):
        return self.all_contours_objects[path_num - 1]
    #endregion
    #region Setters
    def update_section_saturation_and_point(self, direction):
        #TODO: replace this with node saturation
        total_count, (min_y, max_y, min_x, max_x), sub_counts, new_point = (
            self.helper.single_dir_parallels(self.cur_point, self.outer_edges, direction, self.maze_sections))
        for sub_count in sub_counts:
            cur_section = self.maze_sections.sections[(sub_count['y_sec'], sub_count['x_sec'])]
            if 0 <= sub_count['y_sec'] < self.maze_sections.m and 0 <= sub_count['x_sec'] < self.maze_sections.n and not \
                cur_section.saturated:
                cur_section.update_saturation(self.maze_sections, sub_count['sub_count'])

        #Update section if needed
        if len(sub_counts) > 0:
            self.cur_section = self.helper.retrieve_new_section(new_point, self.maze_sections)
            self.set_compass(CompassType.outer_attraction_compass)

        #Update point
        self.cur_point = new_point
        self.path.append(self.cur_point)
        self.path_nd = np.vstack((self.path_nd, np.array(self.cur_point)))
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








