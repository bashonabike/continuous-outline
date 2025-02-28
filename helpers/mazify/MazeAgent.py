import math
import numpy as np

import helpers.mazify.temp_options as options
import helpers.mazify.MazeAgentHelpers as helpers

class MazeAgent:
    def __init__(self, outer_edges, inner_edges, maze_sections):
        self.path = []
        self.outer_edges = outer_edges
        self.inner_edges = inner_edges
        self.maze_sections = maze_sections
        self.helper = helpers.MazeAgentHelpers()

    def proximity_to_edge(self, point):
        #Adding 10E-3 to avoid floating point errors
        # testtt = np.arange(0.0, 2*math.pi - options.directions_incr + 0.001, options.directions_incr)
        # for direction in np.arange(0.0, 2*math.pi - options.directions_incr + 0.001, options.directions_incr):
        proximities = self.helper.process_points_in_quadrant_boxes_to_weighted_centroids(point, self.outer_edges,
                                                                                         options.proximity_search_radius)
        proximities_compass = self.helper.compute_compass_from_quadrant_vectors(proximities)
        return proximities_compass

    def check_intersects(self, point):
        intersects_vectors = self.helper.check_intersects_by_direction(point, self.path)
        intersects_compass = self.helper.compute_compass_from_quadrant_vectors(intersects_vectors)
        return intersects_compass

    def check_parallels(self, point):
        parallels_vectors = self.helper.compute_paralells_quadrant_vectors(point, self.outer_edges)
        parallels_compass = self.helper.compute_compass_from_quadrant_vectors(parallels_vectors)
        return parallels_compass

    def update_section_saturation(self, point, direction):
        total_count, (min_y, max_y, min_x, max_x), sub_counts = self.helper.single_dir_parallels(point, self.outer_edges,
                                                                                           direction, self.maze_sections)
        for sub_count in sub_counts:
            if (sub_count['y_sec'], sub_count['x_sec']) in self.maze_sections.sections:
                (self.maze_sections.sections[(sub_count['y_sec'], sub_count['x_sec'])]
                 .update_saturation(sub_count['sub_count']))







