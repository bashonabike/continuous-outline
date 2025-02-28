import math
import numpy as np

import helpers.mazify.temp_options as options
import helpers.mazify.MazeAgentHelpers as helpers
import helpers.mazify.NetworkInputs as inputs

class MazeAgent:
    def __init__(self, outer_edges, inner_edges, maze_sections):
        self.path = []
        self.outer_edges = outer_edges
        self.inner_edges = inner_edges
        self.dims = (outer_edges.shape[0], outer_edges.shape[1])
        self.maze_sections = maze_sections
        self.helper = helpers.MazeAgentHelpers()
        self.cur_section = None
        self.cur_point = (0, 0)
        self.prev_direction = -1
        self.inst_directions = []

        self.legality_compass = None
        self.proximity_compass = None
        self.intersects_compass = None
        self.outer_attraction_compass = None
        self.parallels_compass = None
        self.deflection_compass = None
        self.inner_attraction = 0.0
        self.compasses = [self.legality_compass, self.proximity_compass, self.intersects_compass,
                          self.outer_attraction_compass, self.parallels_compass, self.deflection_compass,
                          self.inner_attraction]


        self.compass_normalizer = 0.0

        self.network_inputs = inputs.NetworkInputs()
        self.build_inputs()

    #region Build
    def build_inputs(self):
        for compass in self.compasses:
            if isinstance(compass, dict):
                for compass_dir in compass.keys():
                    self.network_inputs.add_input( compass[compass_dir], compass_dir)
            else:
                self.network_inputs.add_input(compass, None)
    #endregion
    #region Run
    def run_round(self):
        self.set_direction_vectors()

    def find_compasses(self):
        #Determine raw compasses
        self.legality_compass = self.legality_check()
        self.proximity_compass = self.proximity_to_edge()
        self.intersects_compass = self.check_intersects()
        #NOTE: Outer attraction refreshes only when entering new section
        if self.outer_attraction_compass is None:
            self.outer_attraction_compass = self.check_outer_attraction()
        self.parallels_compass = self.check_parallels()
        self.deflection_compass = self.check_deflection()
        self.inner_attraction = self.check_inner_attraction()

        #Find normalizer
        compass_points_flat = []
        for compass in self.compasses:
            if isinstance(compass, dict):
                compass_points_flat.extend(compass.values())
            else:
                compass_points_flat.append(compass)
        self.compass_normalizer = max(compass_points_flat)

        #Set inputs
        for compass in self.compasses:
            if isinstance(compass, dict):
                for compass_dir in compass.keys():
                    cur_input = self.network_inputs.find_input(compass, compass_dir)
                    cur_input.set_value(compass[compass_dir]/self.compass_normalizer)
            else:
                cur_input = self.network_inputs.find_input(compass, None)
                cur_input.set_value(compass/self.compass_normalizer)

    def find_start_point(self):
        #Find start section
        edge_pixels_array= np.vectorize(lambda x: x.edge_pixels)(self.maze_sections.sections)
        max_edges_index = np.argmax(edge_pixels_array)
        start_maze_section = self.maze_sections.sections.flat[max_edges_index]

        #Start point in section one max clustered
        start_point = start_maze_section.cluster_point_abs
        return start_point
    #endregion
    #region Compassing
    def legality_check(self):
        legal_compass = self.helper.legality_compass(self.inst_directions)
        return legal_compass
    def proximity_to_edge(self):
        #Adding 10E-3 to avoid floating point errors
        proximities = self.helper.process_points_in_quadrant_boxes_to_weighted_centroids(self.cur_point, self.outer_edges,
                                                                                         options.proximity_search_radius)
        proximities_compass = self.helper.compute_compass_from_quadrant_vectors(proximities)
        return proximities_compass


    def check_intersects(self):
        intersects_compass = self.helper.check_intersects_by_direction_compass(self.inst_directions, self.path)
        return intersects_compass

    def check_parallels(self):
        parallels_compass = self.helper.compute_paralells_compass(self.cur_point, self.inst_directions,
                                                                  self.outer_edges)
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
    #region Setters
    def update_section_saturation_and_point(self, direction):
        total_count, (min_y, max_y, min_x, max_x), sub_counts, new_point = (
            self.helper.single_dir_parallels(self.cur_point, self.outer_edges, direction, self.maze_sections))
        for sub_count in sub_counts:
            if (sub_count['y_sec'], sub_count['x_sec']) in self.maze_sections.sections:
                (self.maze_sections.sections[(sub_count['y_sec'], sub_count['x_sec'])]
                 .update_saturation(sub_count['sub_count']))

        #Update section if needed
        if len(sub_counts) > 0:
            self.cur_section = self.helper.retrieve_new_section(new_point, self.maze_sections)
            self.outer_attraction_compass =self.check_outer_attraction()

        #Update point
        self.cur_point = new_point
        self.path.append(self.cur_point)
        self.prev_direction = direction

    def set_direction_vectors(self):
        self.inst_directions = self.helper.parse_direction_vector_starters(self.cur_point, self.dims)
    #endregion








