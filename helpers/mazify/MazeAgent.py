import math
from logging import exception

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d
import pandas as pd

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
        self.all_contours_objects, self.outer_contours_objects, self.inner_contours_objects  = [], [], []
        for i in range(len(self.all_contours)):
            new_path = EdgePath(i + 1, self.all_contours[i], maze_sections, i < len(self.outer_contours))
            self.all_contours_objects.append(new_path)
            if new_path.outer: self.outer_contours_objects.append(new_path)
            else: self.inner_contours_objects.append(new_path)

        self.update_distances_by_contour()

        maze_sections.set_dumb_nodes()



        self.dims = (outer_edges.shape[0], outer_edges.shape[1])
        self.maze_sections = maze_sections
        self.helper = helpers.MazeAgentHelpers()
        self.cur_section = None
        self.cur_point, self.cur_node = (0, 0), None
        self.start_node, self.end_node = None, None
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
        self.start_node, cur_point, cur_section = self.find_start_node()
        self.end_node, end_point, end_section = self.find_end_node(self.start_node)

        section_path = shorty.find_shortest_path_with_coverage(self.maze_sections.dumb_nodes_weighted,
                                                            self.maze_sections.dumb_nodes_req,
                                                            (cur_section.y_sec, cur_section.x_sec),
                                                            (end_section.y_sec, end_section.x_sec))

        raw_path = self.draw_raw_path(section_path)
        raw_path_coords = [n.point for n in raw_path]
        return raw_path_coords
        # self.set_direction_vectors()
        # self.set_compasses(on_edge=True)
        # self.prev_direction, _ = getdir.get_direction(self.network_inputs, on_edge=True)
        # image = Image.open(im_orig_path)
        #
        # while not self.maze_sections.check_saturation():
        #     self.walk_edge_until_exit_section()
        #     self.cur_node = None
        #     while self.cur_node is None:
        #         self.set_direction_vectors()
        #         self.set_compasses(off_edge=True)
        #         direction, _ = getdir.get_direction(self.network_inputs, off_edge=True)
        #         #TODO: Configure with staying power factored in
        #         self.cur_node = self.check_intersect_edge_update_point(direction)
        #     self.plot_path(self.path, image) #TEMPPP

    def update_distances_by_contour(self):
        for path in self.all_contours_objects:
            #Determine dist from all points
            num_trackers = len(path.section_tracker)
            all_dists = np.zeros((num_trackers, num_trackers), dtype=np.uint16)
            dumb_weight = options.dumb_node_required_weight if path.outer else options.dumb_node_optional_weight
            mid_point = len(path.section_tracker)//2
            rows, cols = all_dists.shape
            i_indices, j_indices = np.indices((rows, cols))
            #Shift around midpoint since items can at most be 1/2 apart (it loops around)
            all_dists[:] = (dumb_weight*(mid_point - np.abs(i_indices - j_indices - mid_point)%num_trackers))
            direct_paths = (path.section_tracker[i_indices], path.section_tracker[j_indices])

            #Coalesce to find min dist for each section
            all_sections_coded = [options.maze_sections_across*t.section.y_sec + t.section.x_sec
                                  for t in path.section_tracker]
            df_dists = pd.DataFrame(all_dists, index=all_sections_coded, columns=all_sections_coded)
            df_paths = pd.DataFrame(direct_paths, index=all_sections_coded, columns=all_sections_coded)

            #NOTE!!!! This stores ALL paths need to match on the min

            def min_coalesce(group):
                return group.min()

            def concatenate_tuples(series):
                """Concatenates tuples in a Pandas Series into a list."""
                result = []
                for item in series:
                    if isinstance(item, tuple):
                        result.append(item)
                return result

            coalesced_dists_df = df_dists.groupby(level=0).apply(lambda x: x.groupby(level=1, axis=1).
                                                                 apply(min_coalesce))
            coalesced_paths_df = df_paths.groupby(level=0).apply(lambda x: x.groupby(level=1, axis=1).
                                                                 apply(concatenate_tuples))


            # Reorder rows and columns by label order
            coalesced_dists_df = coalesced_dists_df.sort_index().sort_index(axis=1)
            coalesced_paths_df = coalesced_paths_df.sort_index().sort_index(axis=1)

            # Extract sorted labels and data as ndarray
            #TODO: Make sure sorting by int not string!!  May need to prepad with 0's of digits of max val
            sorted_sections_coded = coalesced_dists_df.index.to_numpy()
            coalesced_dists_nd = coalesced_dists_df.to_numpy()
            coalesced_paths_nd = coalesced_paths_df.to_numpy()

            #
            # unique_labels = sorted(list(set(path.section_tracker)))
            # label_to_index = {label: i for i, label in enumerate(unique_labels)}
            #
            # num_unique = len(unique_labels)
            # coalesced_weights = np.full((num_unique, num_unique), np.inf)  # Initialize with infinity
            #
            # for i, row_label in enumerate(path.section_tracker):
            #     for j, col_label in enumerate(path.section_tracker):
            #         row_idx = label_to_index[row_label]
            #         col_idx = label_to_index[col_label]
            #         coalesced_weights[row_idx, col_idx] = min(coalesced_weights[row_idx, col_idx], all_dists[i, j])

            # Calculate boolean mask of updates
            update_mask = (coalesced_dists_nd <
                           self.maze_sections.dumb_nodes_distances[sorted_sections_coded[:, None], sorted_sections_coded])

            # Use advanced indexing and minimum update
            self.maze_sections.dumb_nodes_distances[sorted_sections_coded[:, None], sorted_sections_coded] = np.minimum(
                self.maze_sections.dumb_nodes_distances[sorted_sections_coded[:, None], sorted_sections_coded],
                coalesced_dists_nd
            )

            #Update paths where neccessary
            self.maze_sections.dumb_nodes_distances_trackers_path[sorted_sections_coded[:, None],
            sorted_sections_coded][update_mask] = coalesced_paths_nd[update_mask]

    def draw_raw_path(self, section_path: list):
        #TODO: use path bits for connections even if doesnt connect to any other seg, just use part of path look at proxim
        #to others

        #TODO: compute distance between sections when doing path walk! check each grouping if shortest then update matrix,
        #also update matrix of paths (do paths as list of sectioon_trackers)
        #Make sure to do flip version across identity line each time, reverse section_tracker path

        #Then for each node with an edge (outer or inner) that has no path defined or path weight > euclidian dist*factor,
        #Need to join together, do divide and conquer search for nearest neighbour that has a distance (if 3 squares away there is
        # a node with a short dist to the given node, use it, else break up into multiple segments)
        #MAYBE TRY TO FIND C/C++ LIB THAT WILL BUILD DIST FROM PARTIAL DIST MATRIX
        #I guess only need to find dists between req nodes.  if doing divide n conquer may need full matrix pop tho
        #find all connecties within bounding box formed as 1 outside each point, maybe have min width & height
        #Store all connecties as weighted edges, then push all connectie-end points into ND array, linalg with self,
        # find all 0 < dist < 2, make sure filter out dupl since bi-directional (ndarray way do this?? combinatorics?)
        #Push them in as weight of bridging, djikstra it all
        #Use these section_trackers to find path from start to end
        #If not inflection, can block-copy nodes from section-tracker to path (may need to reverse it, pre-cache reversed nodes)
        #If inflection, block-copy portions of nodes list

        # Get paths for each section
        prev_section, prev_path =  self.start_node.section, self.start_node.path
        pathified_section_path = []
        for i in range(len(section_path)):
            if i == 24:
                sdfd=""
            cur_section = self.maze_sections.sections[section_path[i]]
            if i < len(section_path) - 1:
                next_section = self.maze_sections.sections[section_path[i + 1]]
            else:
                next_section = None
            cur_path, cur_trackers = self.find_path_for_walk_section(prev_section, cur_section, next_section, prev_path)
            pathified_section_path.append({"section": cur_section, "path_set": cur_path, "trackers_sets": cur_trackers})
            prev_section, prev_path = cur_section, cur_path[1]

        #Walk pathified tour
        cur_node, path_rev = self.start_node, False
        raw_path_nodes = []
        start_mode = True
        for i in range(len(pathified_section_path)):
            if i == 24:
                sdfd=""
            pathified_section = pathified_section_path[i]
            if i < len(pathified_section_path) - 1: next_section = pathified_section_path[i + 1]["section"]
            else: next_section = None
            cur_section, cur_path_set, cur_trackers_set = (pathified_section["section"], pathified_section["path_set"],
                                                           pathified_section["trackers_sets"])
            cur_section_tracker = cur_node.section_tracker

            #If start mode, find best direction from start node
            if start_mode and cur_trackers_set is not None and  cur_section_tracker in [t[0] for t in cur_trackers_set]:
                try:
                    path_rev = self.find_path_walk_dir(cur_section_tracker, next_section)
                except:
                    foo = "bar"
                else:
                    start_mode = False

            if cur_trackers_set is not None and cur_path_set[0] is not None and cur_path_set[0] == cur_path_set[1]:
                #Check if need to jump across two trackers
                if cur_section_tracker not in [t[0] for t in cur_trackers_set]:
                    #First one in list since is ordered by shortest distance
                    jump_in_node_set, jump_out_node_set = cur_section_tracker.nodes, cur_trackers_set[0][0].nodes
                    path_jump_nodes = self.find_inflection_points(jump_in_node_set, jump_out_node_set)

                    #If start mode, get dir
                    if start_mode:
                        path_rev = self.find_node_to_node_walk_dir(cur_node, path_jump_nodes[0])
                        start_mode = False

                    # Walk prescribed path until hit end point
                    walk_counter = 0
                    while cur_node is not path_jump_nodes[0]:
                        raw_path_nodes.append(cur_node)
                        cur_node = cur_node.walk(path_rev)
                        walk_counter += 1
                        if walk_counter > 10000: raise exception("Dead loop walkin")
                    raw_path_nodes.append(cur_node)

                    # Set up 2nd part of path
                    cur_node = path_jump_nodes[1]
                    cur_section_tracker = cur_node.section_tracker
                    try:
                        path_rev = self.find_path_walk_dir(cur_section_tracker, next_section)
                    except:
                        sdf=""

                #Walk the path until hit next section
                walk_counter = 0
                while ((next_section is None and (cur_node.section == cur_section and cur_node is not self.end_node)) or
                       (next_section is not None and cur_node.section != next_section)):
                    raw_path_nodes.append(cur_node)
                    cur_node = cur_node.walk(path_rev)
                    walk_counter += 1
                    if walk_counter > 10000: raise exception("Dead loop walkin")
                if next_section is None and cur_node is self.end_node: raw_path_nodes.append(cur_node)
            elif cur_trackers_set is not None and cur_path_set[0] != cur_path_set[1]:
                #If inflection point, find best spot to jump paths
                inflec_nodes = (None, None)
                if cur_path_set[0] is not None:
                    in_nodes = cur_section_tracker.nodes
                else:
                    in_nodes = [raw_path_nodes[-1]]

                if cur_path_set[1] is not None:
                    out_nodes = cur_trackers_set[0][0].nodes
                elif next_section is not None:
                    out_nodes = []
                    for j in range(i + 1, len(pathified_section_path)):
                        test_section_path = pathified_section_path[j]
                        if test_section_path["path_set"][1] is not None and test_section_path["trackers_sets"] is not None:
                            out_nodes = test_section_path["trackers_sets"][0][0].nodes
                            if len(out_nodes) == 0: raise exception("Path is broken, investigate")
                            break
                else:
                    out_nodes = [cur_section_tracker.get_out_node(path_rev)]

                inflec_nodes = list(self.find_inflection_points(in_nodes, out_nodes))

                #If start mode, get dir
                if start_mode:
                    path_rev = self.find_node_to_node_walk_dir(cur_node, inflec_nodes[0])
                    start_mode = False

                #Set none as needed so as not to double-lap next section
                if cur_path_set[0] is None: inflec_nodes[0] = None
                if cur_path_set[1] is None: inflec_nodes[1] = None

                if cur_path_set[0] is not None:
                    #Walk prescribed path until hit end point
                    walk_counter = 0
                    while cur_node is not inflec_nodes[0]:
                        raw_path_nodes.append(cur_node)
                        cur_node = cur_node.walk(path_rev)
                        walk_counter += 1
                        if walk_counter > 10000: raise exception("Dead loop walkin")
                    raw_path_nodes.append(cur_node)

                if cur_path_set[1] is not None:
                    #Lock to new path
                    cur_node = inflec_nodes[1]
                    cur_section_tracker = cur_node.section_tracker

                    #Determine best direction
                    try:
                        path_rev = self.find_path_walk_dir(cur_section_tracker, next_section)
                    except:
                        sdfsd=""

                    #Walk the path until hit next section
                    walk_counter = 0
                    while ((next_section is None and (cur_node.section == cur_section and cur_node is not self.end_node))
                           or (next_section is not None and cur_node.section != next_section)):
                        raw_path_nodes.append(cur_node)
                        cur_node = cur_node.walk(path_rev)
                        walk_counter += 1
                        if walk_counter > 10000: raise exception("Dead loop walkin")
                    if next_section is None and cur_node is self.end_node: raw_path_nodes.append(cur_node)

        return raw_path_nodes

    def find_path_walk_dir(self, inst_section_tracker, next_section):
        fwd_moves, rev_moves = 0, 0
        for rev in [False, True]:
            temp_tracker = inst_section_tracker
            for m in range(options.section_tracker_max_walk + 2):
                temp_tracker = temp_tracker.get_next_tracker(rev)
                if not rev:
                    fwd_moves += 1
                else:
                    rev_moves += 1
                if temp_tracker.section == next_section: break
        if fwd_moves > options.section_tracker_max_walk and rev_moves > options.section_tracker_max_walk:
            raise exception("Couldnt do the walk")
        path_rev = rev_moves < fwd_moves
        return path_rev

    def find_node_to_node_walk_dir(self, node1, node2):
        #NOTE: this assumes nodes are in same section and on same path + path section tracker
        for rev in [False, True]:
            cur_node = node1
            while cur_node.section_tracker == node1.section_tracker:
                cur_node = cur_node.walk(rev)
                if cur_node is node2: return rev

        raise exception("Couldnt do the walk node to node")



    def check_where_can_walk_path_between_sections(self, path, section1, section2):
        section1_indices = np.where(path.section_tracker_red_nd_doubled == section1)[0]
        section2_indices = np.where(path.section_tracker_red_nd_doubled == section2)[0]

        # Calculate distances
        distances = np.abs(section1_indices[:, np.newaxis] - section2_indices[np.newaxis, :])

        # Find minimum distance and indices
        min_indices = np.argmin(distances, axis=1)

        # Filter out distances greater than the threshold
        min_distances = distances[np.arange(len(section1_indices)), min_indices]
        within_threshold = min_distances <= options.section_tracker_max_walk
        suitable_walk_indices = np.vstack((np.arange(len(section1_indices))[within_threshold],
                                           min_indices[within_threshold]))
        suitable_distances = min_distances[within_threshold]
        suitable_walks_nd = (np.array([section1_indices[suitable_walk_indices[0]],
                                   section2_indices[suitable_walk_indices[1]]]))
        suitable_walks_nd %= (len(path.section_tracker_red_nd_doubled) // 2)
        suitable_walks_distances_nd = np.vstack((suitable_walks_nd, suitable_distances)).T
        suitable_walks_redund = suitable_walks_distances_nd.tolist()
        suitable_walks_tup = tuple([(l[0], l[1], l[2]) for l in suitable_walks_redund])
        suitable_walks = list(set(suitable_walks_tup))
        suitable_walks.sort(key=lambda x: x[2])

        if len(suitable_walks) > 0:
            # Retrieve closest tracker section nums
            return ([(path.section_tracker[w[0]],path.section_tracker[w[1]], w[2])
                     for w in suitable_walks])
        else:
            return None


    def find_inflection_points(self, path_1_seg, path_2_seg):
        points_1, points_2 = np.array([n.point for n in path_1_seg]), np.array([n.point for n in path_2_seg])

        if not points_1.size or not points_2.size:
            return None, None # Handle empty paths

        # Calculate distances
        distances = np.linalg.norm(points_1[:, np.newaxis, :] - points_2[np.newaxis, :, :], axis=2)

        # Find minimum distance and indices
        min_indices = np.unravel_index(np.argmin(distances), distances.shape)

        # Retrieve closest points
        return path_1_seg[min_indices[0]], path_2_seg[min_indices[1]]


    def find_indices(self, my_list, value):
        """
        Finds the indices where a value exists in a list using list comprehension.

        Args:
            my_list: The input list.
            value: The value to search for.

        Returns:
            A list of indices where the value is found.
        """
        return [index for index, item in enumerate(my_list) if item == value]

    def find_path_for_walk_section(self, prev_section, cur_section, next_section, prev_path):
        #TOOD: Maybe follow edges within section even if does not join to oth section
        in_path, out_path = None, None

        #If prev_section exists, we know that prev_path exists in this section (if it exists)
        if prev_section is not None: in_path = prev_path

        #No next_section, out_path N/A
        if next_section is None:
            return (in_path, self.end_node.path), None

        #Try to find prev_path in next section
        if in_path is not None and in_path in next_section.paths:
            #Check to see if/where path connects regions without significant wandering
            trackers = self.check_where_can_walk_path_between_sections(in_path, cur_section, next_section)
            if trackers is not None:
                return (in_path, in_path), trackers

        #Otherwise find new one.  Check outers first, inners if needed for continuous path
        for sections_type in [1, 2]:
            if sections_type == 1:
                cand_paths = cur_section.outer_paths & next_section.outer_paths
            else:
                cand_paths = cur_section.inner_paths & next_section.inner_paths

            for cand_path in cand_paths:
                #Check to see if/where path connects regions without significant wandering
                trackers = self.check_where_can_walk_path_between_sections(cand_path, cur_section, next_section)
                if trackers is not None:
                    out_path = cand_path
                    return (in_path, out_path), trackers

        #Default cannot find path out
        return (in_path, None), None

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

    def find_end_node(self, start_node: EdgeNode):
        #Find a point approx across
        start_point = start_node.point
        approx_end_point = ((self.dims[0] - 1) - start_point[0], (self.dims[1] - 1) - start_point[1])
        nearest_outer_point = self.find_closest_outer_edge_point(approx_end_point)
        end_node = self.find_closest_node_to_edge_point(nearest_outer_point)
        return end_node, end_node.point, end_node.section

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








