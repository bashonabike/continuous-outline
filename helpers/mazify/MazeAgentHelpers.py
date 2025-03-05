import numpy as np
from numpy.ma.core import indices
from shapely.geometry import LineString, Point, MultiPoint
import math
import time

import helpers.mazify.temp_options as options
from helpers.Enums import CompassDir
import helpers.mazify.MazeSections as sections
from helpers.mazify.EdgeNode import EdgeNode

#TODO: use LineString.Simplify to simplify path

class MazeAgentHelpers:

    def __init__(self):
        self.sin_lut, self.parallel_sin_lut, self.segment_sin_lut = self.create_sin_lut()
        self.cos_lut, self.parallel_cos_lut, self.segment_cos_lut = self.create_cos_lut()

    #region LUTs
    def create_sin_lut(self):
        angles = np.arange(0.0, 2 * math.pi - options.directions_incr + 0.001, options.directions_incr)
        sin_values = np.sin(angles)
        parallel_sin_values = options.parallel_search_radius * sin_values
        segment_sin_values = options.segment_length * sin_values
        return sin_values, parallel_sin_values, segment_sin_values

    def create_cos_lut(self):
        angles = np.arange(0.0, 2 * math.pi - options.directions_incr + 0.001, options.directions_incr)
        cos_values = np.cos(angles)
        parallel_cos_values = options.parallel_search_radius * cos_values
        segment_cos_values = options.segment_length * cos_values
        return cos_values, parallel_cos_values, segment_cos_values

    def approx_sin(self, angle_rad):
        index = int(angle_rad * len(self.sin_lut) / (2 * np.pi)) % len(self.sin_lut)
        return self.sin_lut[index], self.parallel_sin_lut[index], self.segment_sin_lut[index]

    def approx_cos(self, angle_rad):
        index = int(angle_rad * len(self.cos_lut) / (2 * np.pi)) % len(self.cos_lut)
        return self.cos_lut[index], self.parallel_cos_lut[index], self.segment_cos_lut[index]
    #endregion
    #region Misc Helpers
    def bound_coords(self, y, x, dims):
        y_mod = round(min(max(y, 0), dims[0] - 1), 0)
        x_mod = round(min(max(x, 0), dims[1] - 1), 0)
        return int(y_mod), int(x_mod)
    #endregion
    #region Inputs
    def process_points_in_quadrant_boxes_to_weighted_centroids(self,point, image, radius):
        """
        Finds True points within quadrant bounding boxes, computes centroids,
        and creates vectors pointing to centroids.

        Args:
            point: A tuple (y, x) representing the point's coordinates.
            image: A 2D NumPy array (binary image) with True/False values.
            radius: The search radius, defining the bounding box size.

        Returns:
            A list of tuples (centroid_y, centroid_x, vector_y, vector_x).
        """

        y, x = point
        rows, cols = image.shape
        results = []

        # Define quadrant bounding boxes
        quadrant_boxes = {
            1: (max(0, int(y - radius)), int(y), int(x), min(cols, int(x + radius + 1))),  # Top-right
            2: (max(0, int(y - radius)), int(y), max(0, int(x - radius)), int(x)),  # Top-left
            3: (int(y), min(rows, int(y + radius + 1)), max(0, int(x - radius)), int(x)),  # Bottom-left
            4: (int(y), min(rows, int(y + radius + 1)), int(x), min(cols, int(x + radius + 1)))  # Bottom-right
        }

        for quadrant, (min_y, max_y, min_x, max_x) in quadrant_boxes.items():
            # Find points within quadrant bounding box
            found_points = self.find_true_points_in_bbox(image, (min_y, max_y, min_x, max_x))

            if found_points.size > 0:
                centroid_y = np.mean(found_points[:, 0])
                centroid_x = np.mean(found_points[:, 1])
                distance = np.sqrt((centroid_y - y)**2 + (centroid_x - x)**2)

                # Avoid division by zero
                vector_y = (centroid_y - y) / (distance + 1.0)
                vector_x = (centroid_x - x) / (distance + 1.0)
                results.append({'quadrant': quadrant, 'cent_y': centroid_y, 'cent_x': centroid_x, 'vec_y': vector_y,
                                'vec_x': vector_x})
            else:
                results.append({'quadrant': quadrant, 'cent_y': 0.0, 'cent_x': 0.0, 'vec_y': 0.0, 'vec_x': 0.0})
        return results

    def compute_compass_from_quadrant_vectors(self,quadrant_vectors):
        """
       Compute compass directions for input to network

       Args:
           quadrant_vectors: list(dict).

       Returns:
           Dict of compass directions
       """
        compass = {CompassDir.N:0.0, CompassDir.E:0.0, CompassDir.S:0.0, CompassDir.W:0.0}
        for quadrant in quadrant_vectors:
            if quadrant['quadrant'] in (1, 2):
                compass[CompassDir.N] += abs(quadrant['vec_y'])
            else:
                compass[CompassDir.S] += abs(quadrant['vec_y'])

            if quadrant['quadrant'] in (1, 4):
                compass[CompassDir.E] += abs(quadrant['vec_x'])
            else:
                compass[CompassDir.W] += abs(quadrant['vec_x'])

        return compass

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Return true if line segments AB and CD intersect
    def intersect(self, vec1, vec2):
        return (self.ccw(vec1[0], vec2[0], vec2[1]) != self.ccw(vec1[1], vec2[0], vec2[1]) and
                self.ccw(vec1[0], vec1[1], vec2[0]) != self.ccw(vec1[0], vec1[1], vec2[1]))

    def check_intersections(self,segment_proposed, segments_existing: list, centroids_existing: np.array):
        """
        Finds path indices within a bounding box, groups neighbors, and checks for intersections.

        Args:
            segment_proposed: Tuple of tuples ((x1, y1), (x2, y2)) representing the line segment.
            segments_existing: List of tuple coordinates (x, y).

        Returns:
            Number of intersects found
        """
        if len(segments_existing) == 0: return 0

        num_intersections = 0

        dx = dy = options.segment_length
        centroid = tuple((segment_proposed[0][i] + segment_proposed[1][i]) // 2 for i in range(2))
        min_y, min_x = centroid[0] - dy, centroid[1] - dx
        max_y, max_x = centroid[0] + dy, centroid[1] + dx

        if centroids_existing.size > 2:
            bb_mask = (
                (centroids_existing[:, 0] >= min_y) &
                (centroids_existing[:, 0] <= max_y) &
                (centroids_existing[:, 1] >= min_x) &
                (centroids_existing[:, 1] <= max_x)
            )
            indices_in_bbox = np.where(bb_mask)[0].tolist()
        else:
            bb_mask = (
                    (centroids_existing[0] >= min_y) &
                    (centroids_existing[0] <= max_y) &
                    (centroids_existing[1] >= min_x) &
                    (centroids_existing[1] <= max_x)
            )
            indices_in_bbox = [0] if bb_mask else []

        for i in indices_in_bbox:
            if self.intersect(segments_existing[i], segment_proposed):
                num_intersections += 1

        return num_intersections

    def parse_direction_vector_starters(self, point, dims):
        direction_vectors = []
        for direction in np.arange(0.0, 2 * math.pi - options.directions_incr + 0.001, options.directions_incr):
            #NOTE: flip y since origin is top left
            new_tent_y_raw = round(point[0] - self.approx_sin(direction)[2], 0)
            new_tent_x_raw = round(point[1] + self.approx_cos(direction)[2], 0)
            new_tent_y, new_tent_x = int(new_tent_y_raw), int(new_tent_x_raw)

            tent_line = [(point[0], point[1]), (new_tent_y, new_tent_x)]
            norm_vector = np.array([new_tent_y - point[0], new_tent_x - point[1]])/options.segment_length
            if new_tent_y < 0 or new_tent_y >= dims[0] or new_tent_x < 0 or new_tent_x >= dims[1]:
                legal = 0
            else:
                legal = 1
            cur_dir = {'quadrant': 1 + direction//(math.pi/2), 'direction': direction, 'new_tent_y': new_tent_y,
                       'new_tent_x': new_tent_x, 'tent_line': tent_line, 'norm_vector': norm_vector, 'legal': legal,
                       'inst_weight': 0.0}
            direction_vectors.append(cur_dir)
        return direction_vectors

    def parse_weighted_dir_vectors_into_compass(self, direction_vectors):
        """
       Compute compass directions for input to network

       Args:
           direction_vectors: list(dict).

       Returns:
           Dict of compass directions
       """
        compass = {CompassDir.N:0.0, CompassDir.E:0.0, CompassDir.S:0.0, CompassDir.W:0.0}
        for direction_vector in direction_vectors:
            weighted_vector = direction_vector['inst_weight'] * direction_vector['norm_vector']
            if direction_vector['quadrant'] in (1, 2):
                compass[CompassDir.N] += abs(weighted_vector[0])
            else:
                compass[CompassDir.S] += abs(weighted_vector[0])

            if direction_vector['quadrant'] in (1, 4):
                compass[CompassDir.E] += abs(weighted_vector[1])
            else:
                compass[CompassDir.W] += abs(weighted_vector[1])
        return compass

    def check_intersects_by_direction_compass(self, direction_vectors, segments_existing: list,
                                              centroids_existing: np.array):
        for direction_vector in direction_vectors:
            intersects = self.check_intersections(direction_vector['tent_line'],
                                                  segments_existing, centroids_existing)
            direction_vector['inst_weight'] = 1.0/(1 + math.pow(intersects, 2))
        compass = self.parse_weighted_dir_vectors_into_compass(direction_vectors)
        return compass

    def count_edge_pixels_paralleled(self, edges, start_point, end_point, direction, sub_box=None):
        left_ortho = (direction - math.pi/2) % (2 * math.pi)
        right_ortho = (direction + math.pi/2) % (2 * math.pi)
        #TODO: if too crappy, try to approx diagonally
        #TODO: Make this faster!
        min_y, max_y, min_x, max_x = 99999, -1, 99999, -1
        dims = edges.shape

        #Determine bounding box
        for dir in [left_ortho, right_ortho]:
            for point in [start_point, end_point]:
                # NOTE: flip y since origin is top left
                y_raw = point[0] - self.approx_sin(dir)[1]
                x_raw = point[1] + self.approx_cos(dir)[1]
                y, x = self.bound_coords(y_raw, x_raw, dims)

                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)

        #If sub_box is provided, use it
        if sub_box is not None:
            min_y = max(min_y, sub_box[0])
            max_y = min(max_y, sub_box[1])
            min_x = max(min_x, sub_box[2])
            max_x = min(max_x, sub_box[3])

        #Determine saturation of box
        sub_array = edges[min_y:max_y, min_x:max_x]
        return np.sum(sub_array), (min_y, max_y, min_x, max_x)

    def compute_parallels_compass(self, point, direction_vectors, edges):
        for direction_vector in direction_vectors:
            count, _ = self.count_edge_pixels_paralleled(edges, point,
                                                         (direction_vector['new_tent_y'],
                                                          direction_vector['new_tent_x']),
                                                         direction_vector['direction'])
            direction_vector['inst_weight'] = count
        compass = self.parse_weighted_dir_vectors_into_compass(direction_vectors)
        return compass

    def get_next_point(self, point, direction, edges, maze_sections: sections.MazeSections):
        # NOTE: flip y since origin is top left
        new_tent_y_raw = point[0] - self.approx_sin(direction)[2]
        new_tent_x_raw = point[1] + self.approx_cos(direction)[2]
        new_tent_y, new_tent_x = self.bound_coords(new_tent_y_raw, new_tent_x_raw, edges.shape)

        #Check new section
        new_section = maze_sections.get_section_from_coords(new_tent_y, new_tent_x)

        return new_section, (new_tent_y, new_tent_x)




    def single_dir_parallels(self, point, edges, direction, maze_sections):
        # NOTE: flip y since origin is top left
        new_tent_y_raw = point[0] - self.approx_sin(direction)[2]
        new_tent_x_raw = point[1] + self.approx_cos(direction)[2]
        new_tent_y, new_tent_x = self.bound_coords(new_tent_y_raw, new_tent_x_raw, edges.shape)
        count, (min_y, max_y, min_x, max_x) = self.count_edge_pixels_paralleled(edges, point, (new_tent_y,
                                                                                               new_tent_x), direction)

        # Check if crossing over into mult sections
        min_y_sec, min_x_sec = maze_sections.get_section_indices_from_coords(min_y, min_x)
        max_y_sec, max_x_sec = maze_sections.get_section_indices_from_coords(max_y, max_x)
        sub_counts = []
        for y_sec in range(min_y_sec, max_y_sec + 1):
            for x_sec in range(min_x_sec, max_x_sec + 1):
                if 0 <= y_sec < maze_sections.m and 0 <= x_sec < maze_sections.n:
                    bbox = (maze_sections.y_grade*y_sec, maze_sections.y_grade*(y_sec + 1),
                            maze_sections.x_grade*x_sec, maze_sections.x_grade* (x_sec + 1))
                    sub_count, _ = self.count_edge_pixels_paralleled(edges, point,
                                                                     (new_tent_y, new_tent_x),
                                                                     direction, bbox)
                    sub_counts.append({'y_sec': y_sec, 'x_sec': x_sec, 'sub_count': sub_count})


        return count, (min_y, max_y, min_x, max_x), sub_counts, (new_tent_y, new_tent_x)

    def saturation_quadrant_vectors(self, cur_quadrant, maze_sections):
        return 1

    def legality_compass(self, direction_vectors):
        for direction_vector in direction_vectors:
            direction_vector['inst_weight'] = direction_vector['legal']
        compass = self.parse_weighted_dir_vectors_into_compass(direction_vectors)
        return compass

    def deflection_compass(self, direction_vectors, prev_direction):
        for direction_vector in direction_vectors:
            if (prev_direction > -1 and
                    (options.max_deflect_rad < abs(direction_vector['direction'] - prev_direction)
                    < options.rev_max_deflect_rad)):
                direction_vector['inst_weight'] = 0
            else: direction_vector['inst_weight'] = 1
        compass = self.parse_weighted_dir_vectors_into_compass(direction_vectors)
        return compass

    def outer_sections_attraction_compass(self, maze_sections, cur_section):
        compass = {CompassDir.N:0.0, CompassDir.E:0.0, CompassDir.S:0.0, CompassDir.W:0.0}
        for row in maze_sections.sections:
            for section in row:
                if section is cur_section: continue

                dy, dx = section.y_sec - cur_section.y_sec, section.x_sec - cur_section.x_sec
                normalizer = max(abs(dx), abs(dy))
                dy_norm, dx_norm = dy/normalizer, dx/normalizer
                attraction = self.inner_section_attraction_scalar(section)

                if dy_norm < 0:
                    #reverse since image origin is top left
                    compass[CompassDir.N] += abs(dy_norm)*attraction
                else:
                    compass[CompassDir.S] += abs(dy_norm)*attraction

                if dx_norm > 0:
                    compass[CompassDir.E] += abs(dx_norm)*attraction
                else:
                    compass[CompassDir.W] += abs(dx_norm)*attraction

        return compass

    def inner_section_attraction_scalar(self, cur_section):
        return cur_section.attraction


    def edge_magnetism_scalar(self, cur_node:EdgeNode, edge_rev:bool, maze_sections: sections.MazeSections):
        cur_path, cur_section, cur_section_tracker_num = cur_node.path, cur_node.section, cur_node.section_tracker_num

        #Look ahead prescribed num sections, check how many are satisfied
        if not edge_rev:
            inst_section_tracker = cur_path.section_tracker
            inst_tracker_num = cur_section_tracker_num
        else:
            inst_section_tracker = cur_path.section_tracker_rev
            inst_tracker_num = (len(cur_path.section_tracker) - 1) - cur_section_tracker_num

        section_tracker_end = inst_tracker_num + options.edge_magnetism_look_ahead_sections
        section_tracker_split = (len(inst_section_tracker) - section_tracker_end)
        if section_tracker_split >= 0:
            look_ahead_sections = inst_section_tracker[inst_tracker_num:section_tracker_end]
        else:
            look_ahead_sections = (inst_section_tracker[inst_tracker_num:len(inst_section_tracker)] +
            inst_section_tracker[0:(-1)*section_tracker_split])

        distinct_sections = list(set(look_ahead_sections))
        magnetism = sum([s.attraction for s in distinct_sections if not s.saturated])

        return magnetism


    #endregion
    #region Getters

    def retrieve_new_section(self, point, maze_sections: sections.MazeSections):
        y_sec_new, x_sec_new = maze_sections.get_section_indices_from_coords(point[0], point[1])
        if 0 <= y_sec_new < maze_sections.m and 0 <= x_sec_new < maze_sections.n:
            return maze_sections.sections[y_sec_new, x_sec_new]

        return None
    #endregion
    #region Misc Helpers
    def find_true_points_in_bbox(self,array, bbox):
        """
        Finds all True points within a bounding box in a NumPy ndarray.

        Args:
            array: The NumPy ndarray (boolean).
            bbox: A tuple (min_y, max_y, min_x, max_x) defining the bounding box.

        Returns:
            A NumPy array of (y, x) coordinates of True points.
        """

        min_y, max_y, min_x, max_x = bbox
        sub_array = array[min_y:max_y, min_x:max_x]
        y_indices, x_indices = np.where(sub_array)

        # Adjust indices to the original array's coordinates
        y_indices += min_y
        x_indices += min_x

        return np.column_stack((y_indices, x_indices))
    #endregion