import math
import numpy as np

import helpers.mazify.temp_options as options
import helpers.mazify.MazeAgentHelpers as helpers

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

    #region Run
    def run_round(self):
        self.set_direction_vectors()

    def normalized_compasses(self):
        #TODO: Normalize to the greatest direction of all compasses
        pass

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
            #TODO: re-calc section sat compass

        #Update point
        self.cur_point = new_point
        self.path.append(self.cur_point)
        self.prev_direction = direction

    def set_direction_vectors(self):
        self.inst_directions = self.helper.parse_direction_vector_starters(self.cur_point, self.dims)
    #endregion








