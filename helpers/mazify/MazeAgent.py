from logging import exception
# import itertools
import numpy as np
# from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt
# from scipy.signal import convolve2d

# import helpers.mazify.temp_options as options
import helpers.mazify.MazeSections as sections
# from helpers.Enums import TraceTechnique
from helpers.mazify.EdgePath import EdgePath
# from helpers.mazify.EdgeNode import EdgeNode
import helpers.mazify.NetworkxExtension as nxex

#TODO: Also check if focal regions or expl control points changed since last rev, if so regen sections agent (split focus masks level between)

class MazeAgent:
    def __init__(self, options, outer_edges, outer_contours, inner_edges, inner_contours,
                 maze_sections: sections.MazeSections, from_db=False, all_contours_objects:list=None,
                max_tracker_size=-1, start_node=None, end_node=None):
        self.options = options
        #NOTE: edges are codified numerically to correspond with outer contours
        self.outer_edges, self.outer_contours = outer_edges, outer_contours
        self.inner_edges, self.inner_contours = inner_edges, inner_contours
        self.maze_sections = maze_sections

        self.all_edges_bool, self.all_contours = (np.where(self.outer_edges + self.inner_edges > 0, True, False),
                                             self.outer_contours + self.inner_contours)

        if not from_db:
            self.all_contours_objects, self.outer_contours_objects, self.inner_contours_objects  = [], [], []
            if len(self.inner_contours) > 0:
                max_inner_contour_len = max([len(contour) for contour in self.inner_contours])
            else:
                max_inner_contour_len = 0
            self.max_tracker_size = 0
            for i in range(len(self.all_contours)):
                new_path = EdgePath(options, i + 1, self.all_contours[i], maze_sections, i < len(self.outer_contours),
                                    max_inner_contour_len)
                self.all_contours_objects.append(new_path)
                if new_path.outer: self.outer_contours_objects.append(new_path)
                else: self.inner_contours_objects.append(new_path)
                if len(new_path.section_tracker) > self.max_tracker_size:
                    self.max_tracker_size = len(new_path.section_tracker)

            self.maze_sections.set_section_node_cats()
            self.start_node, self.end_node = None, None
        else:
            self.all_contours_objects, self.outer_contours_objects, self.inner_contours_objects = (all_contours_objects,
                                                                                                   [], [])
            self.outer_contours_objects = [p for p in self.all_contours_objects if p.outer]
            self.inner_contours_objects = [p for p in self.all_contours_objects if not p.outer]
            self.outer_contours_objects.sort(key=lambda p: p.num)
            self.inner_contours_objects.sort(key=lambda p: p.num)
            self.max_tracker_size = max_tracker_size
            self.start_node, self.end_node = start_node, end_node

        self.dims = (outer_edges.shape[0], outer_edges.shape[1])
        self.cur_section = None
        self.cur_point, self.cur_node = (0, 0), None
        self.edge_rev = False

    @classmethod
    def from_df(cls, options, outer_edges, outer_contours, inner_edges, inner_contours,
                 maze_sections: sections.MazeSections, all_contours_objects:list,
                max_tracker_size, start_node, end_node):
        return cls(options, outer_edges, outer_contours, inner_edges, inner_contours,
                 maze_sections, from_db=True, all_contours_objects=all_contours_objects,
                   max_tracker_size=max_tracker_size, start_node=start_node, end_node=end_node)

    def run_round_trace_approx_path(self,  parent_inkex, approx_ctrl_points_nd:np.array):
        #Find inflection points
        inflection_points = []
        if self.options.trace_inner_too:
            search_sects = [s.coords_sec for s in self.maze_sections.sections.flatten().tolist()
                            if s.dumb_req or s.dumb_opt]
        else:
            outer_trackers = [t for trackers in [c.section_tracker for c in self.outer_contours_objects]
                              for t in trackers]
            search_sects = list(set([t.section.coords_sec for t in outer_trackers]))
        sectionized_ctrl_points = approx_ctrl_points_nd//np.array((self.maze_sections.y_grade,
                                                                   self.maze_sections.x_grade))
        for ctrl_sec, orig_pt in zip(sectionized_ctrl_points.tolist(), approx_ctrl_points_nd.tolist()):
            closest_sec_coords = self.find_closest_sect(ctrl_sec, search_sects)
            closest_sec = self.maze_sections.sections[closest_sec_coords[0], closest_sec_coords[1]]
            #Find closest point in section
            _, closest_node = self.find_inflection_points([orig_pt], closest_sec.nodes,
                                                          path_1_expl_point=True)
            node_path_idx = closest_node.num
            closest_node_idx_in_tracker = node_path_idx - closest_node.section_tracker.in_node.num
            closest_graph_node = (closest_sec_coords[0], closest_sec_coords[1],
                                  closest_node.path_num, closest_node.section_tracker_num)

            inflection_points.append({"orig_pt": orig_pt,
                                      "closest_sec": closest_sec,
                                      "closest_sec_coords": closest_sec_coords,
                                      "closest_node": closest_node,
                                      "closest_node_idx_in_tracker": closest_node_idx_in_tracker,
                                      "closest_pt": closest_node.point,
                                      "closest_graph_node": closest_graph_node})
            parent_inkex.msg(f"{ctrl_sec} -> {inflection_points[-1]['closest_sec_coords']}")
            parent_inkex.msg(f"{orig_pt} -> {inflection_points[-1]['closest_pt']}")

        #TODO: Ensure that closest point is always traversed?  May not be neccesary

        # Determine best path for rough trace
        section_path = []
        for t in range(len(inflection_points) - 1):
            section_path.append(inflection_points[t]['closest_graph_node'] +
                                (inflection_points[t]['closest_node_idx_in_tracker'],))
            section_path.extend(nxex.shortest_path(self.maze_sections.path_graph,
                                                   inflection_points[t]['closest_graph_node'],
                                                   inflection_points[t + 1]['closest_graph_node'],
                                                   weight='weight'))
        section_path.append(inflection_points[-1]['closest_graph_node'] +
                            (inflection_points[-1]['closest_node_idx_in_tracker'],))

        # Trace path from tracker path
        nodes_path = self.set_node_path_from_sec_path(section_path)
        raw_path_coords = [n.point for n in nodes_path]
        return raw_path_coords, section_path, approx_ctrl_points_nd.tolist()

    def set_node_path_from_sec_path(self, sections_nodes_path):
        nodes_path = []
        jump_path = False
        prev_sect, cur_sect, next_sect = None, None, None
        omit_repeat_sect = None
        must_hit_node_idx_pre, must_hit_node_idx_post = None, None
        cur_tracker = None
        cur_tracker_idx = -1
        prev_len = 0
        path_rev = False
        inflection_out = None
        for i in range(len(sections_nodes_path)):
            cur_sect = sections_nodes_path[i]
            if i < len(sections_nodes_path) - 1:
                next_sect = sections_nodes_path[i + 1]
            else:
                next_sect = None

            if len(cur_sect) == 5 and i == 0:
                must_hit_node_idx_pre = cur_sect[4] #Start point
            elif len(cur_sect) == 4:
                if omit_repeat_sect is not None and omit_repeat_sect == cur_sect:
                    continue
                omit_repeat_sect = None #NOTE: keep this here in case multiple repeats of same section

                #Peek ahead for must-hit point
                if next_sect is not None and len(next_sect) == 5:
                    must_hit_node_idx_post = next_sect[4]

                if inflection_out is not None:
                    cur_tracker_idx = inflection_out['tracker_path_idx']
                elif must_hit_node_idx_pre is not None:
                    cur_tracker_idx = must_hit_node_idx_pre
                else:
                    cur_tracker_idx = -1
                inflection_out = None
                cur_tracker = self.all_contours_objects[cur_sect[2] - 1].section_tracker[cur_sect[3]]
                path_closed = cur_tracker.path.closed

                if next_sect is not None and len(next_sect) == 4:
                    #Tracker identified, check if prev sect lines up
                    trackers_in_path = len(cur_tracker.path.section_tracker)
                    tracker_modulo = trackers_in_path if path_closed else 999999
                    if (next_sect[2] != cur_sect[2] or cur_sect[3] not in
                        ((next_sect[3] + 1)%tracker_modulo, (next_sect[3] - 1)%tracker_modulo)):
                        raise exception("Something screwy with tracking")
                    path_rev = (cur_sect[3] - 1)%tracker_modulo == next_sect[3]
                    if path_rev:
                        if cur_tracker_idx == -1:
                            nodes_path.extend(list(reversed(cur_tracker.nodes)))
                        else:
                            nodes_path.extend(list(reversed(cur_tracker.nodes[:cur_tracker_idx + 1])))
                    else:
                        if cur_tracker_idx == -1:
                            nodes_path.extend(cur_tracker.nodes)
                        else:
                            nodes_path.extend(cur_tracker.nodes[cur_tracker_idx:])
                elif next_sect is not None and len(next_sect) == 5:
                    #Must hit point identified, we know the cur sect is a tracker
                    if cur_tracker_idx != -1:
                        #Check direction, make sure moving from cur_tracker_idx to must_hit_node_idx_post
                        path_rev = must_hit_node_idx_post < cur_tracker_idx
                    if path_rev:
                        if cur_tracker_idx == -1:
                            nodes_path.extend(list(reversed(cur_tracker.nodes[must_hit_node_idx_post:])))
                        else:
                            nodes_path.extend(list(reversed(cur_tracker.nodes[must_hit_node_idx_post:cur_tracker_idx + 1])))
                    else:
                        if cur_tracker_idx == -1:
                            nodes_path.extend(cur_tracker.nodes[:must_hit_node_idx_post + 1])
                        else:
                            nodes_path.extend(cur_tracker.nodes[cur_tracker_idx:must_hit_node_idx_post + 1])
                elif next_sect is not None:
                    #Entering into the abyss, look for next intersection
                    next_trackerized_sect, sections_path_idx_new = None, -1
                    for j in range(i + 2, len(sections_nodes_path)):
                        if len(sections_nodes_path[j]) == 4:
                            # if sections_nodes_path[j] == sections_nodes_path[i]:
                            #     omit_repeat_sect = sections_nodes_path[j]
                            # else:
                            next_trackerized_sect = sections_nodes_path[j]
                            sections_path_idx = j
                            break

                    if next_trackerized_sect is not None:
                        if next_trackerized_sect != cur_sect:
                            #Find best intersection point
                            #TODO: If the next on is part of same path just keep going
                            #TODO: Better handling of not doing weird loopdy loops
                            next_tracker = self.all_contours_objects[next_trackerized_sect[2] - 1]. \
                                section_tracker[next_trackerized_sect[3]]
                            inflec_nodes = list(self.find_inflection_points(cur_tracker.nodes,
                                                                            next_tracker.nodes,
                                                                            retrieve_idxs=True))
                            if path_rev:
                                if cur_tracker_idx == -1:
                                    nodes_path.extend(list(reversed(cur_tracker.nodes[inflec_nodes[0]:])))
                                else:
                                    nodes_path.extend(list(reversed(cur_tracker.nodes[inflec_nodes[0]:cur_tracker_idx + 1])))
                            else:
                                if cur_tracker_idx == -1:
                                    nodes_path.extend(cur_tracker.nodes[:inflec_nodes[0]:])
                                else:
                                    nodes_path.extend(cur_tracker.nodes[cur_tracker_idx:inflec_nodes[0] + 1])

                            inflection_out = {"sections_path_idx": sections_path_idx, "tracker_path_idx": inflec_nodes[1]}

                        else:
                            midpoint = len(cur_tracker.nodes)//2
                            if path_rev:
                                if cur_tracker_idx == -1:
                                    nodes_path.extend(list(reversed(cur_tracker.nodes[midpoint:])))
                                else:
                                    if midpoint >= cur_tracker_idx: midpoint = max(0, cur_tracker_idx - 1)
                                    nodes_path.extend(list(reversed(cur_tracker.nodes[midpoint:cur_tracker_idx + 1])))
                            else:
                                if cur_tracker_idx == -1:
                                    nodes_path.extend(cur_tracker.nodes[:midpoint])
                                else:
                                    if midpoint <= cur_tracker_idx: midpoint = min(len(cur_tracker.nodes) - 1,
                                                                                   cur_tracker_idx + 1)
                                    nodes_path.extend(cur_tracker.nodes[cur_tracker_idx:midpoint + 1])

                            inflection_out = {"sections_path_idx": sections_path_idx, "tracker_path_idx": midpoint}
                else:
                    #Run out the last section
                    if path_rev:
                        if cur_tracker_idx == -1:
                            nodes_path.extend(list(reversed(cur_tracker.nodes)))
                        else:
                            nodes_path.extend(list(reversed(cur_tracker.nodes[:cur_tracker_idx + 1])))
                    else:
                        if cur_tracker_idx == -1:
                            nodes_path.extend(cur_tracker.nodes)
                        else:
                            nodes_path.extend(cur_tracker.nodes[cur_tracker_idx:])

                #MAKE SURE set dir when non tracker (shouldn't be issue but maybe check)

                must_hit_node_idx_pre, must_hit_node_idx_post = must_hit_node_idx_post, None
        return nodes_path

    def find_inflection_points(self, path_1_seg, path_2_seg, retrieve_idxs=False, path_1_expl_point=False):
        if path_1_expl_point:
            points_1 = np.array(path_1_seg)
        else:
            points_1 = np.array([n.point for n in path_1_seg])
        points_2 = np.array([n.point for n in path_2_seg])

        if not points_1.size or not points_2.size:
            return None, None # Handle empty paths

        # Calculate distances
        distances = np.linalg.norm(points_1[:, np.newaxis, :] - points_2[np.newaxis, :, :], axis=2)

        # Find minimum distance and indices
        min_indices = np.unravel_index(np.argmin(distances), distances.shape)

        # Retrieve closest points
        if not retrieve_idxs:
            return path_1_seg[min_indices[0]], path_2_seg[min_indices[1]]
        else:
            return min_indices[0], min_indices[1]

    def find_closest_sect(self, match_point, cand_sects, return_idx=False):
        points_1, points_2 = np.array([match_point]), np.array(cand_sects)

        if not points_1.size or not points_2.size:
            return None  # Handle empty paths

        # Calculate distances
        distances = np.linalg.norm(points_1[:, np.newaxis, :] - points_2[np.newaxis, :, :], axis=2)

        # Find minimum distance and indices
        min_indices = np.unravel_index(np.argmin(distances), distances.shape)

        # Retrieve closest points
        if not return_idx:
            return cand_sects[min_indices[1]]
        else:
            return min_indices[1]
    #endregion
    #region Walkies
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
    #endregion








