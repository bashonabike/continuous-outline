import math
import numpy as np

import helpers.mazify.temp_options as options
import helpers.mazify.maze_agent_helpers as helpers

class MazeAgent:
    def __init__(self, edges):
        self.path = []
        self.edges = edges

    def proximity_to_edge(self, point):
        #Adding 10E-3 to avoid floating point errors
        # testtt = np.arange(0.0, 2*math.pi - options.directions_incr + 0.001, options.directions_incr)
        # for direction in np.arange(0.0, 2*math.pi - options.directions_incr + 0.001, options.directions_incr):
        proximities = helpers.process_points_in_quadrant_boxes_to_weighted_centroids(point, self.edges,
                                                                                     options.proximity_search_radius)
        proximities_compass = helpers.compute_compass_from_quadrant_vectors(proximities)
        return proximities_compass






