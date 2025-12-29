from logging import exception
# import itertools
import numpy as np
import random as rd
# from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt
# from scipy.signal import convolve2d

# import helpers.mazify.temp_options as options
import helpers.mazify.MazeSections as sections
# from helpers.Enums import TraceTechnique
from helpers.mazify.EdgePath import EdgePath
# from helpers.mazify.EdgeNode import EdgeNode
import helpers.mazify.NetworkxExtension as nxex


# TODO: Also check if focal regions or expl control points changed since last rev, if so regen sections agent (split focus masks level between)

class MazeAgent:
    def __init__(self, parent_inkex, options, outer_edges, outer_contours, inner_edges, inner_contours,
                 maze_sections: sections.MazeSections, from_db=False, all_contours_objects: list = None,
                 max_tracker_size=-1, start_node=None, end_node=None):
        """
        Initialize a MazeAgent instance.

        Args:
            parent_inkex: Reference to the parent Inkscape extension instance.
            options: Configuration options for the path.
            outer_edges: Pixel map of the outer edges.
            outer_contours: List of contours of the outer edges.
            inner_edges: Pixel map of the inner edges.
            inner_contours: List of contours of the inner edges.
            maze_sections: MazeSections instance for section management.
            from_db: Whether this path is being loaded from a database. Defaults to False.
            all_contours_objects: List of EdgePath objects, if not None.
            max_tracker_size: Maximum length of inner contours for weight calculation.
            start_node: Node to start the path from, if not None.
            end_node: Node to end the path at, if not None.

        Returns:
            None
        """
        self.options = options
        # NOTE: edges are codified numerically to correspond with outer contours
        self.outer_edges, self.outer_contours = outer_edges, outer_contours
        self.inner_edges, self.inner_contours = inner_edges, inner_contours
        self.maze_sections = maze_sections

        self.all_edges_bool, self.all_contours = (np.where(self.outer_edges + self.inner_edges > 0, True, False),
                                                  self.outer_contours + self.inner_contours)

        if not from_db:
            self.all_contours_objects, self.outer_contours_objects, self.inner_contours_objects = [], [], []
            if len(self.inner_contours) > 0:
                max_inner_contour_len = max([len(contour) for contour in self.inner_contours])
            else:
                max_inner_contour_len = 0
            self.max_tracker_size = 0
            for i in range(len(self.all_contours)):
                new_path = EdgePath(parent_inkex, options, i + 1, self.all_contours[i], maze_sections,
                                    i < len(self.outer_contours),
                                    max_inner_contour_len)
                self.all_contours_objects.append(new_path)
                if new_path.outer:
                    self.outer_contours_objects.append(new_path)
                else:
                    self.inner_contours_objects.append(new_path)
                if len(new_path.section_tracker) > self.max_tracker_size:
                    self.max_tracker_size = len(new_path.section_tracker)

            self.maze_sections.set_section_node_cats()
            self.maze_sections.set_direct_jump_close_nodes(parent_inkex)
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
                maze_sections: sections.MazeSections, all_contours_objects: list,
                max_tracker_size, start_node, end_node):
        """
        Construct a MazeAgent from a DataFrame.

        Args:
            options: Configuration options for the path.
            outer_edges: Pixel map of the outer edges.
            outer_contours: List of contours for the outer edges.
            inner_edges: Pixel map of the inner edges.
            inner_contours: List of contours for the inner edges.
            maze_sections: MazeSections instance for section management.
            all_contours_objects: List of edgePath objects.
            max_tracker_size: Maximum length of inner contours for weight calculation.
            start_node: Node to start the path from, if not None.
            end_node: Node to end the path at, if not None.

        Returns:
            MazeAgent instance.
        """
        return cls(options, outer_edges, outer_contours, inner_edges, inner_contours,
                   maze_sections, from_db=True, all_contours_objects=all_contours_objects,
                   max_tracker_size=max_tracker_size, start_node=start_node, end_node=end_node)

    def run_round_trace_approx_path(self, parent_inkex, approx_ctrl_points_nd: np.array, max_magnet_lock_dist):
        # Find inflection points
        """
        Runs a round-trace approx path generation algorithm.

        Finds inflection points and determines the best path for a rough trace.
        Trims unneccessary blips and concatenates the path as needed.
        Inserts jump node where direct jump is required.
        Rebuilds without shortcuts.

        Args:
            parent_inkex: The Inkex object that the path will be rendered in.
            approx_ctrl_points_nd: The approximate control points in (y, x) format.

        Returns:
            A list of (y, x) points representing the path.
            A list of (y, x) points representing the path with shortcuts removed.
            A list of (y, x) points representing the approximate control points.
            A list of (y, x) points representing the node trace coordinates.
            A list of dictionaries containing information about the trace path in SVG.
        """
        inflection_points = []
        if self.options.trace_inner_too:
            search_sects = [s.coords_sec for s in self.maze_sections.sections.flatten().tolist()
                            if s.dumb_req or s.dumb_opt]
        else:
            outer_trackers = [t for trackers in [c.section_tracker for c in self.outer_contours_objects]
                              for t in trackers]
            search_sects = list(set([t.section.coords_sec for t in outer_trackers]))
        sectionized_ctrl_points = (approx_ctrl_points_nd // np.array((self.maze_sections.y_grade,
                                                                      self.maze_sections.x_grade))).astype(int)
        sectionized_ctrl_points = np.maximum(0, sectionized_ctrl_points)
        sectionized_ctrl_points = np.minimum(int(self.options.maze_sections_across - 1), sectionized_ctrl_points)
        search_grid_width = int(1 + 2 * ((2 * max_magnet_lock_dist) // (self.maze_sections.y_grade +
                                                                        self.maze_sections.x_grade)))
        prev_lock_node = None
        node_trace_coord_path = []
        test_output_trace_coords = []
        for ctrl_sec, orig_pt in zip(sectionized_ctrl_points.tolist(), approx_ctrl_points_nd.tolist()):
            # Find closest point in section with locking if appl
            closest_sec, closest_node = self.find_suitable_inflec_node(parent_inkex, ctrl_sec, orig_pt, search_sects,
                                                                       prev_lock_node,
                                                                       search_grid_width,
                                                                       self.options.prefer_outer_contours_locking)
            # if abs(ctrl_sec[0] - closest_sec.coords_sec[0]) + abs(ctrl_sec[1] - closest_sec.coords_sec[1]) > 0:
            # parent_inkex.msg(f"Sec {ctrl_sec} has req {self.maze_sections.sections[ctrl_sec[0], ctrl_sec[1]].dumb_req} opt {self.maze_sections.sections[ctrl_sec[0], ctrl_sec[1]].dumb_opt}")
            test_output_trace_coords.append({"pt": closest_node.point, "sec_orig": str(ctrl_sec)})

            node_trace_coord_path.append(closest_node.point)

            # ####TEMP###
            # import inkex
            # rect_style = 'fill:#000000;stroke:none;stroke-width:0.264583'
            # el2 = inkex.Rectangle.new(closest_sec.xmin, closest_sec.ymin, self.maze_sections.x_grade, self.maze_sections.y_grade)
            # el2.style = rect_style
            # parent_inkex.svg.get_current_layer().add(el2)
            # ###############

            node_path_idx = closest_node.num
            closest_node_idx_in_tracker = node_path_idx - closest_node.section_tracker.in_node.num
            closest_sec_coords = closest_sec.coords_sec
            closest_graph_node = (closest_sec_coords[0], closest_sec_coords[1],
                                  closest_node.path_num, closest_node.section_tracker_num)

            inflection_points.append({"orig_pt": orig_pt,
                                      "closest_sec": closest_sec,
                                      "closest_sec_coords": closest_sec_coords,
                                      "closest_node": closest_node,
                                      "closest_node_idx_in_tracker": closest_node_idx_in_tracker,
                                      "closest_pt": closest_node.point,
                                      "closest_graph_node": closest_graph_node})
            # parent_inkex.msg(f"{ctrl_sec} -> {inflection_points[-1]['closest_sec_coords']}")
            # parent_inkex.msg(f"{orig_pt} -> {inflection_points[-1]['closest_pt']}")

            prev_lock_node = closest_node

        # ###########teeemppppp
        # valid_edges = []
        # all_graph_edges = self.maze_sections.path_graph.edges(data=True)
        # for u, v, data in all_graph_edges:
        #     if 85 <= u[0] <= 100 and 85 <= v[0] <= 100 and 50 <= u[1] <= 70 and 50 <= v[1] <= 70:
        #         valid_edges.append((u, v, data))
        #
        # parent_inkex.msg(valid_edges)

        #     x = "1698.2827175444538"
        #     y = "2437.924523666384"
        #     text = "[95, 57]"
        #     style = "font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:192px;font-family:sans-serif;"
        #     id = "text163" / >
        #
        # < text
        # x = "1898.1068003556309"
        # y = "2441.096334504657"
        # text = "[96, 62]"

        ###########################

        # Determine best path for rough trace
        section_path = []
        for t in range(len(inflection_points) - 1):
            #     section_path.append(inflection_points[t]['closest_graph_node'] +
            #                         (inflection_points[t]['closest_node_idx_in_tracker'],))
            #     section_path.extend(nxex.shortest_path(self.maze_sections.path_graph,
            #                                            inflection_points[t]['closest_graph_node'],
            #                                            inflection_points[t + 1]['closest_graph_node'],
            #                                            weight='weight'))
            # section_path.append(inflection_points[-1]['closest_graph_node'] +
            #                     (inflection_points[-1]['closest_node_idx_in_tracker'],))

            outer_path_segment = None
            if (self.maze_sections.sections[inflection_points[t]['closest_sec_coords'][0],
            inflection_points[t]['closest_sec_coords'][1]].dumb_req
                    and self.maze_sections.sections[inflection_points[t + 1]['closest_sec_coords'][0],
                    inflection_points[t + 1]['closest_sec_coords'][1]].dumb_req):
                # NOTE: If no path possible, will return None
                outer_path_segment = nxex.shortest_path(self.maze_sections.outer_paths_graph,
                                                        tuple(inflection_points[t]['closest_graph_node'][0:2]),
                                                        tuple(inflection_points[t + 1]['closest_graph_node'][0:2]),
                                                        weight='weight', test_only=True)

            explicit_outer_path_tried = False
            if outer_path_segment is not None:
                path_segment = []
                if len(inflection_points[t]['closest_graph_node']) > 2:
                    path_segment.append(inflection_points[t]['closest_graph_node'])
                path_segment.extend(outer_path_segment)
                if len(inflection_points[t + 1]['closest_graph_node']) > 2:
                    path_segment.append(inflection_points[t + 1]['closest_graph_node'])
                explicit_outer_path_tried = True
            else:
                path_segment = nxex.shortest_path(self.maze_sections.path_graph,
                                                  inflection_points[t]['closest_graph_node'],
                                                  inflection_points[t + 1]['closest_graph_node'],
                                                  weight='weight')

            retry_seg = False
            length_cutoff = 3 if explicit_outer_path_tried else 2
            if len(path_segment) > length_cutoff * (abs(inflection_points[t]['closest_graph_node'][0] -
                                                        inflection_points[t + 1]['closest_graph_node'][0]) +
                                                    abs(inflection_points[t]['closest_graph_node'][1] -
                                                        inflection_points[t + 1]['closest_graph_node'][1])):
                # If very long path, retry
                retry_seg = True
            else:
                sections_crossed = [(n[0], n[1]) for n in path_segment if len(n) > 2]
                if len(sections_crossed) > 0 and len(sections_crossed) / len(set(sections_crossed)) > 1.6:
                    # If frequently doubles back on itself, retry
                    retry_seg = True

            if retry_seg:
                # Retry path, test if shorter
                path_test = nxex.shortest_path(self.maze_sections.path_graph,
                                               inflection_points[t]['closest_graph_node'],
                                               inflection_points[t + 1]['closest_graph_node'],
                                               weight='weight', test_only=True)
                if len(path_test) < len(path_segment):
                    nxex.burn_path(self.maze_sections.path_graph, path_test)
                    path_segment = path_test
            elif explicit_outer_path_tried:
                # Only burn in outer path if explicitly tried and successful
                nxex.burn_path(self.maze_sections.path_graph, outer_path_segment)
                nxex.burn_path(self.maze_sections.outer_paths_graph, outer_path_segment)

            section_path.extend(path_segment[:-1])
            # parent_inkex.msg(f"start: {inflection_points[t]['closest_graph_node']} end: {inflection_points[t+1]['closest_graph_node']} path: {path_segment}")
        section_path.append(inflection_points[-1]['closest_graph_node'])

        # prev_sec, cur_sec = None, None
        # shortcuts = []
        # for i in range(len(section_path)):
        #     cur_sec = (section_path[i][0], section_path[i][1])
        #     if prev_sec is not None and cur_sec != prev_sec:
        #         l_bound, u_bound, cur_src_sec, prev_src_sec = i, i, cur_sec, cur_sec
        #         sections_traversed_shared = set()
        #         while l_bound >0 and u_bound < len(section_path) - 1 and abs(u_bound - l_bound) < 30:
        #             u_sec = (section_path[u_bound][0], section_path[u_bound][1])
        #             l_sec = (section_path[l_bound][0], section_path[l_bound][1])
        #             if u_sec == l_sec:
        #                 l_bound -= 1
        #                 u_bound += 1
        #                 prev_src_sec = cur_src_sec
        #                 sections_traversed_shared.add(cur_src_sec)
        #             elif u_sec == prev_src_sec:
        #                 #Allow upper to catch up
        #                 u_bound += 1
        #                 cur_src_sec = l_sec
        #
        #             elif l_sec == prev_src_sec:
        #                 #Allow lower to catch up
        #                 l_bound -= 1
        #                 cur_src_sec = u_sec
        #             else: break
        #
        #         if u_bound - l_bound > 2 and len(sections_traversed_shared) > 1:
        #             shortcuts.append((l_bound + 1, u_bound - 1))
        #             parent_inkex.msg(f"Short cut {l_bound + 1} -> {u_bound - 1}")
        #
        #     prev_sec = cur_sec
        #
        # # Re-build without shortcuts
        # processed_path = section_path[0:shortcuts[0][0] + 1]
        # for i in range(len(shortcuts)):
        #     # NOTE: including the removed boundary so doesn't chop up path too much
        #     startpoint, endpoint = (shortcuts[i][1],
        #                             shortcuts[i + 1][0] + 1 if i < len(shortcuts) - 1 else len(section_path))
        #     processed_path.append((section_path[startpoint][0], section_path[startpoint][1]))
        #
        #     processed_path.extend(section_path[startpoint:endpoint])
        # processed_path.extend(section_path[shortcuts[-1][1]:])

        # section_path = processed_path
        # TODO: Reenable shortcuts??

        # Insert jump node where direct jump
        final_section_path = []
        insert_start = 0
        for i in range(len(section_path) - 1):
            if len(section_path[i]) >= 4 and len(section_path[i + 1]) >= 4 \
                    and (section_path[i][2] != section_path[i + 1][2] or
                         abs(section_path[i][3] - section_path[i + 1][3]) > 1):
                final_section_path.extend(section_path[insert_start:i + 1])
                final_section_path.append((section_path[i][0], section_path[i][1]))
                insert_start = i + 1
        if insert_start <= len(section_path) - 1:
            final_section_path.extend(section_path[insert_start:])

        # parent_inkex.msg(final_section_path)

        # #Trim unneccessary blips
        # blip_max_len = self.options.maze_sections_across//10
        # shortcuts = []
        # for i in range(1, len(section_path) - 1):
        #     if len(section_path[i]) >= 4 and (len(section_path[i - 1]) == 2 or
        #                                       section_path[i - 1][2] != section_path[i][2]):
        #         #Look back, see if deviated from path unneccessarily
        #         for j in range(i - 2, max(0, i - blip_max_len), -1):
        #             if len(section_path[j]) >= 4 and section_path[j][2] == section_path[i][2]:
        #                 section_diff = abs(section_path[j][0] - section_path[i][0]) + \
        #                     abs(section_path[j][1] - section_path[i][1])
        #                 #Check if in neighbouring or same section
        #                 if section_diff <= 1:
        #                     shortcuts.append([j, i])
        #                     parent_inkex.msg(f"Short cut from {section_path[j]} to {section_path[i]}")
        #                     break
        #
        #             #TEST: If same section and no must hit point hit
        #             if len(section_path[j]) >= 4 and abs(section_path[j][0] - section_path[i][0]) + \
        #                     abs(section_path[j][1] - section_path[i][1]) == 0:
        #                 shortcuts.append([j, i])
        #                 parent_inkex.msg(f"Short cut from {section_path[j]} to {section_path[i]}")
        #                 break
        #
        #             if len(section_path[j]) == 5: break #Must hit these points
        #
        # # Conjoin as needed
        # # Sort the index pairs by start index
        # shortcuts.sort(key=lambda x: x[0])
        #
        # conjoined_inouts = []
        # current_pair = shortcuts[0]
        #
        # for pair in shortcuts[1:]:
        #     if pair[0] <= current_pair[1]:  # Overlapping
        #         current_pair[1] = max(current_pair[1], pair[1])  # Extend end index
        #     else:  # No overlap
        #         conjoined_inouts.append(current_pair)
        #         current_pair = pair
        #
        # conjoined_inouts.append(current_pair)  # Add the last pair
        #
        # # Re-build without shortcuts
        # final_section_path = [section_path[0]]
        # processed_path = section_path[0:conjoined_inouts[0][0] + 1]
        # for i in range(len(conjoined_inouts)):
        #     same_tracker = section_path[conjoined_inouts[i][0]][3] == section_path[conjoined_inouts[i][1]][3]
        #     if not same_tracker:
        #         # NOTE: including the removed boundary so doesn't chop up path too much
        #         startpoint, endpoint = (conjoined_inouts[i][1],
        #                                 conjoined_inouts[i + 1][0] + 1 if i < len(conjoined_inouts) - 1 else len(section_path))
        #         processed_path.append((section_path[startpoint][0], section_path[startpoint][1]))
        #     else:
        #         startpoint, endpoint = (conjoined_inouts[i][1] + 1,
        #                                 conjoined_inouts[i + 1][0] + 1 if i < len(conjoined_inouts) - 1 else len(
        #                                     section_path))
        #
        #     processed_path.extend(section_path[startpoint:endpoint])
        #
        # parent_inkex.msg(f"Short cutted path: {processed_path}")
        # Trace path from tracker path
        nodes_path = self.set_node_path_from_sec_path(parent_inkex, final_section_path)
        raw_path_coords = [n.point for n in nodes_path]
        return raw_path_coords, final_section_path, approx_ctrl_points_nd.tolist(), node_trace_coord_path, test_output_trace_coords

    def set_node_path_from_sec_path(self, parent_inkex, sections_nodes_path):
        """
        Set node path from section path.

        Given a list of sections, each represented by a list of nodes, set the node path for each section.

        The node path is built by tracing the path from one section to the next. The path is built by finding the intersection of the two sections and then tracing the path from that intersection to the end of the current section.

        The node path is then reversed and appended to the node path of the previous section.

        The final node path is the concatenation of all the node paths for each section.

        Parameters:
        parent_inkex (Inkex): The parent Inke object.
        sections_nodes_path (list): A list of sections, each represented by a list of nodes.

        Returns:
        list: The final node path.
        """
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
                must_hit_node_idx_pre = cur_sect[4]  # Start point
            elif len(cur_sect) == 4:
                if omit_repeat_sect is not None and omit_repeat_sect == cur_sect:
                    continue
                omit_repeat_sect = None  # NOTE: keep this here in case multiple repeats of same section

                # Peek ahead for must-hit point
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
                    # Tracker identified, check if prev sect lines up
                    trackers_in_path = len(cur_tracker.path.section_tracker)
                    tracker_modulo = trackers_in_path if path_closed else 999999
                    if (next_sect[2] != cur_sect[2] or cur_sect[3] not in
                            ((next_sect[3] + 1) % tracker_modulo, (next_sect[3] - 1) % tracker_modulo)):
                        parent_inkex.msg(cur_sect)
                        raise exception(f"Something screwy with tracking: {cur_sect}")
                    path_rev = (cur_sect[3] - 1) % tracker_modulo == next_sect[3]
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
                    # Must hit point identified, we know the cur sect is a tracker
                    if cur_tracker_idx != -1:
                        # Check direction, make sure moving from cur_tracker_idx to must_hit_node_idx_post
                        path_rev = must_hit_node_idx_post < cur_tracker_idx
                    if path_rev:
                        if cur_tracker_idx == -1:
                            nodes_path.extend(list(reversed(cur_tracker.nodes[must_hit_node_idx_post:])))
                        else:
                            nodes_path.extend(
                                list(reversed(cur_tracker.nodes[must_hit_node_idx_post:cur_tracker_idx + 1])))
                    else:
                        if cur_tracker_idx == -1:
                            nodes_path.extend(cur_tracker.nodes[:must_hit_node_idx_post + 1])
                        else:
                            nodes_path.extend(cur_tracker.nodes[cur_tracker_idx:must_hit_node_idx_post + 1])
                elif next_sect is not None:
                    # Entering into the abyss, look for next intersection
                    next_trackerized_sect, sections_path_idx = None, -1
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
                            # Find best intersection point
                            # TODO: If the next on is part of same path just keep going
                            # TODO: Better handling of not doing weird loopdy loops

                            next_tracker = self.all_contours_objects[next_trackerized_sect[2] - 1]. \
                                section_tracker[next_trackerized_sect[3]]

                            # if sections_path_idx + 1 < len(sections_nodes_path):
                            #     #Determine next tracker direction
                            #     if len(sections_nodes_path[sections_path_idx + 1]) >= 4:
                            #         trackers_in_path = len(next_tracker.path.section_tracker)
                            #         tracker_modulo = trackers_in_path if path_closed else 999999
                            #         next_path_rev = ((next_trackerized_sect[3] - 1) % tracker_modulo ==
                            #                          sections_nodes_path[sections_path_idx + 1][3])
                            #     else:
                            #         next_next_section_idx =sections_nodes_path[sections_path_idx + 1]
                            #         next_fwd_node, next_rev_node = (next_tracker.out_node.walk(),
                            #                                         next_tracker.in_node.walk(True))
                            #         next_fwd_sec, next_rev_sec = None, None
                            #
                            #         if next_fwd_node is not None:
                            #             next_fwd_sec = next_fwd_node.section.coords_sec
                            #         if next_rev_node is not None:
                            #             next_rev_sec = next_rev_node.section.coords_sec
                            #
                            #         if next_rev_node is None:
                            #             next_path_rev = False
                            #         elif next_fwd_node is None:
                            #             next_path_rev = True
                            #         else:
                            #             fwd_score = abs(next_fwd_sec[0] - next_next_section_idx[0]) + \
                            #                 abs(next_fwd_sec[1] - next_next_section_idx[1])
                            #             rev_score = abs(next_rev_sec[0] - next_next_section_idx[0]) + \
                            #                 abs(next_rev_sec[1] - next_next_section_idx[1])
                            #             next_path_rev = rev_score < fwd_score
                            # else:
                            #     next_path_rev = False
                            #
                            # #Based on next node direction, try to minimize nodes hit on inflection
                            #

                            inflec_nodes = list(self.find_inflection_points(cur_tracker.nodes,
                                                                            next_tracker.nodes,
                                                                            retrieve_idxs=True))
                            if path_rev:
                                if cur_tracker_idx == -1:
                                    nodes_path.extend(list(reversed(cur_tracker.nodes[inflec_nodes[0]:])))
                                else:
                                    nodes_path.extend(
                                        list(reversed(cur_tracker.nodes[inflec_nodes[0]:cur_tracker_idx + 1])))
                            else:
                                if cur_tracker_idx == -1:
                                    nodes_path.extend(cur_tracker.nodes[:inflec_nodes[0]:])
                                else:
                                    nodes_path.extend(cur_tracker.nodes[cur_tracker_idx:inflec_nodes[0] + 1])

                            inflection_out = {"sections_path_idx": sections_path_idx,
                                              "tracker_path_idx": inflec_nodes[1]}

                        else:
                            midpoint = len(cur_tracker.nodes) // 2
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
                    # Run out the last section
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

                # MAKE SURE set dir when non tracker (shouldn't be issue but maybe check)

                must_hit_node_idx_pre, must_hit_node_idx_post = must_hit_node_idx_post, None
        return nodes_path

    def find_suitable_inflec_node(self, parent_inkex, ctrl_sec, orig_pt, search_sects, prev_lock_node,
                                  search_grid_width, prefer_outer):
        """
        Finds the closest suitable inflection node from the given control point and section

        Args:
            parent_inkex: The Inkex object that the path will be rendered in.
            ctrl_sec: The control point coordinates in (y, x) format
            orig_pt: The original point coordinates in (y, x) format
            search_sects: A list of sections to search for the closest inflection node
            prev_lock_node: The previous lock node
            search_grid_width: The width of the search grid
            prefer_outer: Whether to prefer the closest outer contour or not

        Returns:
            A tuple containing the closest section and node
        """
        lock_cands = []
        if prev_lock_node is not None or prefer_outer:
            if prev_lock_node is not None:
                prev_path = prev_lock_node.path
            else:
                prev_path = None
            # Build nxn grid around ctrl sec as possibility for lock node, starting on cur sec

            i_src = range(max(0, ctrl_sec[0] - search_grid_width // 2),
                          min(self.options.maze_sections_across - 1, ctrl_sec[0] + search_grid_width // 2))
            i_src = [ctrl_sec[0]] + [i for i in i_src if i != ctrl_sec[0]]
            j_src = range(max(0, ctrl_sec[1] - search_grid_width // 2),
                          min(self.options.maze_sections_across - 1, ctrl_sec[1] + search_grid_width // 2))
            j_src = [ctrl_sec[1]] + [j for j in j_src if j != ctrl_sec[1]]

            best_pathified_dist, best_outer_dist = None, None
            for i in i_src:
                for j in j_src:
                    test_sec = self.maze_sections.sections[i, j]
                    test_trackers = test_sec.section_trackers
                    test_paths = [tracker.path for tracker in test_trackers]
                    sec_dist_manhat = abs(i - ctrl_sec[0]) + abs(j - ctrl_sec[1])
                    if (prev_lock_node is not None and prev_lock_node.path in test_paths and
                            (best_pathified_dist is None or sec_dist_manhat < best_pathified_dist)):
                        pathified_trackers = [tracker for tracker in test_trackers
                                              if tracker.path == prev_lock_node.path]
                        pathified_trackers_nd = np.array([tracker.tracker_num for tracker in pathified_trackers])
                        trackers_dist = np.abs(pathified_trackers_nd - prev_lock_node.section_tracker_num)
                        if prev_lock_node.path.closed:
                            trackers_in_path = len(prev_lock_node.path.section_tracker)
                            trackers_dist = np.minimum(trackers_dist, trackers_in_path - trackers_dist)
                        closest_tracker = pathified_trackers[np.argmin(trackers_dist)]
                        # Take midpoint node
                        closest_node = closest_tracker.nodes[len(closest_tracker.nodes) // 2]
                        lock_cands.append({"type": 1, "node": closest_node,
                                           "sec_dist_manhat": sec_dist_manhat})
                        best_pathified_dist = sec_dist_manhat
                    elif (prefer_outer and test_sec.dumb_req and
                          (best_outer_dist is None or sec_dist_manhat < best_outer_dist)):
                        pathified_trackers = [tracker for tracker in test_trackers if tracker.path.outer]
                        lock_tracker = pathified_trackers[rd.randint(0, len(pathified_trackers) - 1)]
                        # Take midpoint node
                        lock_node = lock_tracker.nodes[len(lock_tracker.nodes) // 2]
                        lock_cands.append({"type": 2, "node": lock_node,
                                           "sec_dist_manhat": sec_dist_manhat})
                        best_outer_dist = sec_dist_manhat

            if len(lock_cands) > 0:
                lock_cands_sorted = sorted(lock_cands, key=lambda x: (x["type"], x["sec_dist_manhat"]))
                lock_node = lock_cands_sorted[0]["node"]
                return lock_node.section, lock_node

        # Default to closest sec and node otherwise
        closest_sec_coords = self.find_closest_sect(ctrl_sec, search_sects)
        closest_sec = self.maze_sections.sections[closest_sec_coords[0], closest_sec_coords[1]]
        _, closest_node = self.find_inflection_points([orig_pt], closest_sec.nodes,
                                                      path_1_expl_point=True)
        return closest_node.section, closest_node

    def find_inflection_points(self, path_1_seg, path_2_seg, retrieve_idxs=False, path_1_expl_point=False):
        """
        Finds the closest inflection points between two paths.

        Args:
            path_1_seg: Segment of path 1 as list of (y, x) coordinates
            path_2_seg: Segment of path 2 as list of (y, x) coordinates
            retrieve_idxs: Whether to return the indices of the closest points or the points themselves
            path_1_expl_point: Whether to treat path_1_seg as a single point instead of a list of points

        Returns:
            If retrieve_idxs is False, returns the closest points on path_1_seg and path_2_seg
            If retrieve_idxs is True, returns the indices of the closest points on path_1_seg and path_2_seg
        """
        if path_1_expl_point:
            points_1 = np.array(path_1_seg)
        else:
            points_1 = np.array([n.point for n in path_1_seg])
        points_2 = np.array([n.point for n in path_2_seg])

        if not points_1.size or not points_2.size:
            return None, None  # Handle empty paths

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
        """
        Finds the closest section to a given point from a list of candidate sections.

        Args:
            match_point: The point to find the closest section to.
            cand_sects: A list of sections to search through.
            return_idx: Whether to return the index of the closest section instead of the section itself.

        Returns:
            The closest section to `match_point` if `return_idx` is False, otherwise the index of the closest section.
        """
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

    # endregion
    # region Walkies
    def find_closest_node_to_edge_point(self, edge_point):
        # Test outer first, then inner
        """
        Finds the closest node to a given edge point.

        Args:
            edge_point: A 2-tuple (y, x) representing the point to find the closest node to.

        Returns:
            The closest node to the given edge point, or None if no such node exists.
        """
        edge_num = self.outer_edges[edge_point]

        if edge_num == 0:
            edge_num = self.inner_edges[edge_point]

        if edge_num == 0:
            # Phantom edge weird stuff but not worth breaking
            return None

        # Retrieve specified edge path and find closest node
        edge_path = self.get_edge_path_by_number(edge_num)
        point_section = self.maze_sections.get_section_from_coords(edge_point[0], edge_point[1])
        nodes = point_section.get_nodes_by_edge_number(edge_num)
        if len(nodes) == 0:
            # Expand search to surrounding sections
            nodes = point_section.get_surrounding_nodes_by_edge__number(self.maze_sections, edge_num)
        if len(nodes) == 0:
            # Phantom, ignore
            return None

        nodes_coords = [(node.y, node.x) for node in nodes]
        nodes_coords_nd = np.array(nodes_coords)
        edge_point_nd = np.array(edge_point)
        diff = nodes_coords_nd - edge_point_nd
        squared_dist = np.sum(diff ** 2, axis=1)
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

    # endregion
    # region Getters
    def get_edge_path_by_number(self, path_num):
        """
        Retrieves an Edge path object based on its path number.

        Args:
            path_num: An integer representing the path number of the edge path.

        Returns:
            EdgePath: An edge path object corresponding with the given path number.
        """
        return self.all_contours_objects[path_num - 1]
    # endregion
    # region Setters
    # endregion
