import numpy as np
from shapely.geometry import LineString
import math

import helpers.mazify.temp_options as options

#TODO: use LineString.Simplify to simplify path

class MazeAgentHelpers:

    def __init__(self):
        self.sin_lut = self.create_sin_lut()
        self.cos_lut = self.create_cos_lut()

    def create_sin_lut(self):
        angles = np.arange(0.0, 2 * math.pi - options.directions_incr + 0.001, options.directions_incr)
        sin_values = np.sin(angles)
        return sin_values

    def create_cos_lut(self):
        angles = np.arange(0.0, 2 * math.pi - options.directions_incr + 0.001, options.directions_incr)
        cos_values = np.cos(angles)
        return cos_values

    def approx_sin(self, angle_rad):
        index = int(angle_rad * len(self.sin_lut) / (2 * np.pi)) % len(self.sin_lut)
        return self.sin_lut[index]
    def approx_cos(self, angle_rad):
        index = int(angle_rad * len(self.cos_lut) / (2 * np.pi)) % len(self.cos_lut)
        return self.cos_lut[index]

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

    def compute_compass_from_quadrant_vectors(self,quadrant_vectors):
        """
       Compute compass directions for input to network

       Args:
           quadrant_vectors: list(dict).

       Returns:
           Dict of compass directions
       """
        compass = {'N':0.0, 'S':0.0, 'E':0.0, 'W':0.0}
        for quadrant in quadrant_vectors:
            if quadrant['quadrant'] in (1, 2):
                compass['N'] += abs(quadrant['vec_y'])
            else:
                compass['S'] += abs(quadrant['vec_y'])

            if quadrant['quadrant'] in (1, 4):
                compass['E'] += abs(quadrant['vec_x'])
            else:
                compass['W'] += abs(quadrant['vec_x'])

        return compass

    def check_intersections(self,segment_proposed, segments_existing):
        """Checks if segment1 intersects any of the segments in the list."""
        num_intersections = 0
        line1 = LineString(segment_proposed)
        for segment2 in segments_existing:
            line2 = LineString(segment2)
            if line1.intersects(line2):
                num_intersections += 1
        return num_intersections

    def check_intersects_by_direction(self,point, path):
        quadrant_vectors = []
        cur_quadrant = {'quadrant': 1, 'vec_y': 0.0, 'vec_x': 0.0}
        for direction in np.arange(0.0, 2 * math.pi - options.directions_incr + 0.001, options.directions_incr):
            #NOTE: flip y since origin is top left
            new_tent_y = (-1)*(point[0] + options.segment_length * self.approx_cos(direction))
            new_tent_x = point[1] + options.segment_length * self.approx_sin(direction)
            tent_line = [(point[0], point[1]), (new_tent_y, new_tent_x)]
            intersects = self.check_intersections(tent_line, path)
            norm_vector = np.array([new_tent_y - point[0], new_tent_x - point[1]])/options.segment_length
            weighted_vector = intersects * norm_vector
            if direction >= cur_quadrant['quadrant'] * (math.pi/2):
                quadrant_vectors.append(cur_quadrant)
                next_quad_num = cur_quadrant['quadrant'] + 1
                cur_quadrant = {'quadrant': next_quad_num,'vec_y': 0.0, 'vec_x': 0.0}

            cur_quadrant['vec_y'] += weighted_vector[0]
            cur_quadrant['vec_x'] += weighted_vector[1]

        quadrant_vectors.append(cur_quadrant)
        return quadrant_vectors

    def count_edge_pixels_paralleled(self, edges, start_point, end_point, direction, sub_box=None):
        count = 0
        left_ortho = (direction - math.pi/2) % (2 * math.pi)
        right_ortho = (direction + math.pi/2) % (2 * math.pi)
        #TODO: if too crappy, try to approx diagonally
        min_y, max_y, min_x, max_x = 99999, -1, 99999, -1

        #Determine bounding box
        for dir in [left_ortho, right_ortho]:
            for point in [start_point, end_point]:
                # NOTE: flip y since origin is top left
                y = round((-1)*(start_point[0] + options.parallel_search_radius * self.approx_cos(left_ortho)), 0)
                x = round(start_point[1] + options.parallel_search_radius * self.approx_sin(left_ortho), 0)
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

    def compute_paralells_quadrant_vectors(self, point, edges):
        quadrant_vectors = []
        cur_quadrant = {'quadrant': 1, 'vec_y': 0.0, 'vec_x': 0.0}
        for direction in np.arange(0.0, 2 * math.pi - options.directions_incr + 0.001, options.directions_incr):
            # NOTE: flip y since origin is top left
            new_tent_y = (-1) * (point[0] + options.segment_length * self.approx_cos(direction))
            new_tent_x = point[1] + options.segment_length * self.approx_sin(direction)
            count, _ = self.count_edge_pixels_paralleled(edges, point, (new_tent_y, new_tent_x), direction)
            norm_vector = np.array([new_tent_y - point[0], new_tent_x - point[1]]) / options.segment_length
            weighted_vector = count * norm_vector
            if direction >= cur_quadrant['quadrant'] * (math.pi / 2):
                quadrant_vectors.append(cur_quadrant)
                next_quad_num = cur_quadrant['quadrant'] + 1
                cur_quadrant = {'quadrant': next_quad_num, 'vec_y': 0.0, 'vec_x': 0.0}

            cur_quadrant['vec_y'] += weighted_vector[0]
            cur_quadrant['vec_x'] += weighted_vector[1]

        quadrant_vectors.append(cur_quadrant)
        return quadrant_vectors

    def single_dir_parallels(self, point, edges, direction, maze_sections):
        # NOTE: flip y since origin is top left
        new_tent_y = (-1) * (point[0] + options.segment_length * self.approx_cos(direction))
        new_tent_x = point[1] + options.segment_length * self.approx_sin(direction)
        count, (min_y, max_y, min_x, max_x) = self.count_edge_pixels_paralleled(edges, point, (new_tent_y,
                                                                                               new_tent_x), direction)

        # Check if crossing over into mult sections
        min_y_sec, max_y_sec = min_y // maze_sections.y_grade, max_y // maze_sections.y_grade
        min_x_sec, max_x_sec = min_x // maze_sections.x_grade, max_x // maze_sections.x_grade
        sub_counts = []
        for y_sec in range(min_y_sec, max_y_sec + 1):
            for x_sec in range(min_x_sec, max_x_sec + 1):
                if (y_sec, x_sec) in maze_sections.sections:
                    bbox = (maze_sections.y_grade*y_sec, maze_sections.y_grade*(y_sec + 1),
                            maze_sections.x_grade*x_sec, maze_sections.x_grade* (x_sec + 1))
                    sub_count, _ = self.count_edge_pixels_paralleled(edges, point,
                                                                     (new_tent_y, new_tent_x),
                                                                     direction, bbox)
                    sub_counts.append({'y_sec': y_sec, 'x_sec': x_sec, 'sub_count': sub_count})


        return count, (min_y, max_y, min_x, max_x), sub_counts

