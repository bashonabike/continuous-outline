import math
import numpy as np

import helpers.mazify.temp_options as options
import helpers.mazify.MazeAgentHelpers as helpers
import helpers.mazify.NetworkInputs as inputs
import helpers.mazify.MazeSections as sections
import helpers.mazify.TestGetDirection as getdir
from helpers.Enums import CompassType, CompassDir

class MazeAgent:
    def __init__(self, outer_edges, inner_edges, maze_sections: sections.MazeSections):
        self.path = []
        self.outer_edges = outer_edges
        self.inner_edges = inner_edges
        self.all_edges = self.outer_edges + self.inner_edges
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
    def run_round_dumb(self):
        self.cur_point, self.cur_section = self.find_start_point()
        self.path.append(self.cur_point)

        while not self.maze_sections.check_saturation():
            self.set_direction_vectors()
            self.set_compasses()
            direction = getdir.get_direction(self.network_inputs)
            self.update_section_saturation_and_point(direction)
            if len(self.path)%10 == 0:
                sdf=""

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
            if self.compasses.get(compass_def['type']) is None or not compass_def['persist']:
                self.compasses[compass_def['type']] = compass_def['instantiate']()

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
        intersects_compass = self.helper.check_intersects_by_direction_compass(self.inst_directions, self.path)
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
    #region Setters
    def update_section_saturation_and_point(self, direction):
        total_count, (min_y, max_y, min_x, max_x), sub_counts, new_point = (
            self.helper.single_dir_parallels(self.cur_point, self.outer_edges, direction, self.maze_sections))
        for sub_count in sub_counts:
            cur_section = self.maze_sections.sections[(sub_count['y_sec'], sub_count['x_sec'])]
            if 0 <= sub_count['y_sec'] < self.maze_sections.m and 0 <= sub_count['x_sec'] < self.maze_sections.n and not \
                cur_section.saturated:
                cur_section.update_saturation(self.maze_sections.sections, sub_count['sub_count'])

        #Update section if needed
        if len(sub_counts) > 0:
            self.cur_section = self.helper.retrieve_new_section(new_point, self.maze_sections)
            self.set_compass(CompassType.outer_attraction_compass)

        #Update point
        self.cur_point = new_point
        self.path.append(self.cur_point)
        self.prev_direction = direction

    def set_direction_vectors(self):
        self.inst_directions = self.helper.parse_direction_vector_starters(self.cur_point, self.dims)
    #endregion








