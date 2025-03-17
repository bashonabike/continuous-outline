from logging import exception
import itertools
import numpy as np
from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# import helpers.mazify.temp_options as options
import helpers.mazify.MazeSections as sections
from helpers.Enums import TraceTechnique
from helpers.mazify.EdgePath import EdgePath
from helpers.mazify.EdgeNode import EdgeNode
import helpers.mazify.NetworkxExtension as nxex

#TODO: Push maze finding func into sep helper, lazy import it at start of each finder method so not loading on ds_load
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
            max_inner_contour_len = max([len(contour) for contour in self.inner_contours])
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

    #endregion
    #region Run
    # def plot_path(self, path_coords, image):
    #     """
    #     Plots a path defined by a list of tuple coordinates.
    #
    #     Args:
    #         path_coords: A list of tuples, where each tuple represents (x, y) coordinates.
    #     """
    #
    #     if not path_coords:
    #         print("Path is empty.")
    #         return
    #
    #     x_coords, y_coords = zip(*path_coords)
    #
    #     plt.imshow(image)  # Display the image
    #     plt.plot(x_coords, y_coords, color='red', linewidth=1, marker='o', markersize=1)  # Plot the path
    #
    #     plt.axis('off')  # Turn off axis labels and ticks
    #     plt.show(block=True)

    def run_round_trace(self, technique:TraceTechnique):
        match technique:
            case TraceTechnique.snake | 2:
                return self.find_trace_section_tour_snake()
            case TraceTechnique.typewriter | 1:
                return self.find_trace_section_tour_typewriter()
            case TraceTechnique.zigzag_typewriter | 3:
                return self.find_trace_section_tour_zigzag_typewriter()
            case TraceTechnique.vertical_zigzag:
                return self.find_trace_section_tour_vertical_zigzag()
            case TraceTechnique.back_forth:
                return self.find_trace_section_tour_back_forth()


            case _:
                return []

    def find_trace_section_tour_snake(self):
        #TODO: Implement start, need to wrap around
        #Find start track
        # start_node, cur_point, cur_section = self.find_start_node()

        #Find inflection points for each detail chunk
        outer_path = self.all_contours_objects[0].section_tracker
        outer_coords = np.array([t.section.coords_sec for t in outer_path])
        inflection_points = []
        for focus_region in self.maze_sections.focus_region_sections:
            if len(focus_region) == 0: continue
            sections_coords = np.atleast_2d([s.coords_sec for s in focus_region])
            closest_indices_outer, closest_indices_focus = (
                self.find_focus_region_incision_point(outer_coords, sections_coords))
            inflection_points.append({"outer_index_midpoint": sum(closest_indices_outer)//2,
                                      "outer_in_index": closest_indices_outer[0],
                                      "outer_out_index": closest_indices_outer[1],
                                      "outer_in_tracker": outer_path[closest_indices_outer[0]],
                                      "focus_in_section": focus_region[closest_indices_focus[0]],
                                      "outer_out_tracker": outer_path[closest_indices_outer[1]],
                                      "focus_out_section": focus_region[closest_indices_focus[1]],
                                      "focus_sections": sections_coords})

        inflection_points.sort(key=lambda p: p['outer_index_midpoint'])
        prev_out_idx = -1
        for p in inflection_points:
            if p['outer_in_index'] <= prev_out_idx: raise exception("Overlapping focus regions, investigate")
            prev_out_idx = p['outer_out_index']

        #Figure out best path for each details chunk
        sections_nodes_path = []
        prev_point_idx = 0
        for p in inflection_points:
            sections_nodes_path.extend([(t.section.coords_sec[0], t.section.coords_sec[1], t.path_num, t.tracker_num)
                                        for t in outer_path[prev_point_idx:p['outer_in_index'] + 1]])

            sections_nodes_path.extend(nxex.shortest_path(self.maze_sections.path_graph,
                                         p['outer_in_tracker'].section.coords_sec, p['focus_in_section'].coords_sec,
                                                  weight='weight')[:-1])

            focus_poly_points = self.find_maximal_polygon_optimized(p['focus_sections'],
                                                                   p['focus_in_section'].coords_sec,
                                                                   p['focus_out_section'].coords_sec,
                                                                    self.options.snake_details_polygon_faces)
            for h in range(len(focus_poly_points) - 1):
                sections_nodes_path.extend(
                    nxex.shortest_path(self.maze_sections.path_graph, focus_poly_points[h], focus_poly_points[h + 1],
                                     weight='weight')[:-1])

            #NOTE: we want to include the last node this time, do not discount
            sections_nodes_path.extend(nxex.shortest_path(self.maze_sections.path_graph, p['focus_out_section'].coords_sec,
                                                  p['outer_out_tracker'].section.coords_sec, weight='weight')[:])

            prev_point_idx = p['outer_out_index']

        #NOTE: we want to include the last node this time, do not discount
        sections_nodes_path.extend([(t.section.coords_sec[0], t.section.coords_sec[1], t.path_num, t.tracker_num)
                                    for t in outer_path[prev_point_idx:len(outer_path)]])

        #Trace path from tracker path
        nodes_path = self.set_node_path_from_sec_path(sections_nodes_path)
        raw_path_coords = [n.point for n in nodes_path]
        return raw_path_coords

    def find_trace_section_tour_typewriter(self):
        #Find typewriter set points
        outer_path = self.all_contours_objects[0].section_tracker
        outer_sects = list(set([t.section.coords_sec for t in outer_path]))
        typewriter_points = []
        line_height_sects = self.options.maze_sections_across//(self.options.typewriter_lines - 1)
        for l in range(self.options.typewriter_lines):
            #Find line start & end points
            left_sect = self.find_closest_sect((l*line_height_sects, 0), outer_sects)
            right_sect = self.find_closest_sect((l*line_height_sects, self.options.maze_sections_across - 1),
                                                outer_sects)
            typewriter_points.extend([left_sect, right_sect])

        #Determine focus section inclusion
        carriage_returns = [[] for _ in range(self.options.typewriter_lines)]
        carriage_return_zigzags = [[] for _ in range(self.options.typewriter_lines)]
        for focus_region in self.maze_sections.focus_region_sections:
            if len(focus_region) == 0: continue
            sections_coords = np.atleast_2d([s.coords_sec for s in focus_region])

            #Determine method of traversal
            min_y, max_y = np.min(sections_coords[:, 0]), np.max(sections_coords[:, 0])
            min_x, max_x = np.min(sections_coords[:, 1]), np.max(sections_coords[:, 1])
            reg_height, reg_width = max_y - min_y, max_x - min_x

            #Find closest carriage return
            mid_height = (max_y + min_y)//2
            carriage_return = mid_height//line_height_sects

            section_coords_ls = [tuple(s) for s in sections_coords.tolist()]

            if reg_height/line_height_sects < self.options.typewriter_traverse_threshold:
                #Traverse in line with typewriting,
                right_insert, left_insert = (self.find_closest_sect((min_y, max_x), section_coords_ls),
                                             self.find_closest_sect((max_y, min_x), section_coords_ls))
                carriage_returns[carriage_return].extend([right_insert, left_insert])
                # typewriter_points = typewriter_points[:2*(carriage_return + 1)] + [right_insert, left_insert] + \
                #                      typewriter_points[2*(carriage_return + 1):]
            else:
                #Zigzag within carriage return
                zigzag = []
                right_1, left_1 = self.find_closest_sect((min_y, max_x), section_coords_ls), (min_y, min_x),
                right_2, left_2 = (max_y, max_x), self.find_closest_sect((max_y, min_x), section_coords_ls)
                zigzag = (right_1, left_1, right_2, left_2)
                #NOTE: append since need to preserve order
                carriage_return_zigzags[carriage_return].append(zigzag)

        #Build focus regions into typewriter points
        typewriter_points_final = []
        for l in range(self.options.typewriter_lines):
            typewriter_points_final.extend(typewriter_points[2*l:2*(l + 1)])
            straights, zigs = carriage_returns[l], carriage_return_zigzags[l]
            return_line = []

            #Sort by x of leftmost desc
            if len(straights) > 0: straights.sort(key=lambda p: p[1], reverse=True)
            if len(zigs) > 0:
                zigs.sort(key=lambda z: z[0][1], reverse=True)

                #Determine best ordering of focus region inclusion
                zig_ends_flat = [p for z in zigs for p in (z[0], z[-1])]
                zigs_aug = [list(z) for z in zigs]
                for s in straights:
                    closest_end = self.find_closest_sect(s, zig_ends_flat, return_idx=True)
                    zig, at_end = closest_end//2, closest_end%2 == 1
                    if not at_end:
                        zigs_aug[zig] = s + zigs_aug[zig]
                    else:
                        zigs_aug[zig] = zigs_aug[zig] + s

                #Formulate final line
                return_line = [p for z in zigs_aug for p in z]
            elif len(straights) > 0:
                return_line = straights

            typewriter_points_final.extend(return_line)

        #Determine best path for typewriter
        section_path = []
        for t in range(len(typewriter_points_final) - 1):
            section_path.extend(nxex.shortest_path(self.maze_sections.path_graph,
                                                 typewriter_points_final[t], typewriter_points_final[t + 1],
                             weight='weight')[:-1])
        section_path.append(typewriter_points_final[-1])

        #Trace path from tracker path
        nodes_path = self.set_node_path_from_sec_path(section_path)
        raw_path_coords = [n.point for n in nodes_path]
        return raw_path_coords

    def find_trace_section_tour_zigzag_typewriter(self):
        # Find typewriter set points
        outer_path = self.all_contours_objects[0].section_tracker
        outer_sects = list(set([t.section.coords_sec for t in outer_path]))
        typewriter_points = []
        line_height_sects = self.options.maze_sections_across // (self.options.zigzag_typewriter_lines - 1)
        for l in range(self.options.zigzag_typewriter_lines):
            # Find line start & end points
            left_sect = self.find_closest_sect((l * line_height_sects, 0), outer_sects)
            right_sect = self.find_closest_sect((l * line_height_sects + line_height_sects//2,
                                                 self.options.maze_sections_across - 1),
                                                outer_sects)
            typewriter_points.extend([left_sect, right_sect])

        # Determine focus section inclusion
        main_zigzags_straights = [[] for _ in range(2*self.options.zigzag_typewriter_lines)]
        main_zigzags_subzigzags = [[] for _ in range(2*self.options.zigzag_typewriter_lines)]
        for focus_region in self.maze_sections.focus_region_sections:
            if len(focus_region) == 0: continue
            sections_coords = np.atleast_2d([s.coords_sec for s in focus_region])

            # Determine method of traversal
            min_y, max_y = np.min(sections_coords[:, 0]), np.max(sections_coords[:, 0])
            min_x, max_x = np.min(sections_coords[:, 1]), np.max(sections_coords[:, 1])
            reg_height, reg_width = max_y - min_y, max_x - min_x

            # Find closest zig or zag
            mid_height = (max_y + min_y) // 2
            zig_or_zag = mid_height // (line_height_sects//2)
            is_zig, is_zag = zig_or_zag % 2 == 0, zig_or_zag % 2 == 1

            section_coords_ls = [tuple(s) for s in sections_coords.tolist()]

            if reg_height / line_height_sects < self.options.zigzag_typewriter_traverse_threshold:
                # Traverse in line with typewriting,
                if is_zig:
                    right_insert, left_insert = (max_y, max_x), (min_y, min_x)
                    main_zigzags_straights[zig_or_zag].extend([left_insert, right_insert])
                else:
                    right_insert, left_insert = (self.find_closest_sect((min_y, max_x), section_coords_ls),
                                                 self.find_closest_sect((max_y, min_x), section_coords_ls))
                    main_zigzags_straights[zig_or_zag].extend([right_insert, left_insert])
            else:
                # Zigzag within carriage return
                subzigzag = []
                if is_zig:
                    left_1, right_1  = (min_y, min_x), self.find_closest_sect((min_y, max_x), section_coords_ls)
                    left_2, right_2  = self.find_closest_sect((max_y, min_x), section_coords_ls), (max_y, max_x)
                    subzigzag = (left_1, right_1, left_2, right_2)
                else:
                    right_1, left_1 = self.find_closest_sect((min_y, max_x), section_coords_ls), (min_y, min_x)
                    right_2, left_2 = (max_y, max_x), self.find_closest_sect((max_y, min_x), section_coords_ls)
                    subzigzag = (right_1, left_1, right_2, left_2)

                # NOTE: append since need to preserve order
                main_zigzags_subzigzags[zig_or_zag].append(subzigzag)

        # Build focus regions into typewriter points
        zigzag_typewriter_points_final = []
        for l in range(2*self.options.zigzag_typewriter_lines):
            zigzag_typewriter_points_final.extend(typewriter_points[l:l + 1])
            straights, zigs = main_zigzags_straights[l], main_zigzags_subzigzags[l]
            main_zigorzag_line = []

            # Sort by x of leftmost desc
            if len(straights) > 0: straights.sort(key=lambda p: p[1], reverse=True)
            if len(zigs) > 0:
                zigs.sort(key=lambda z: z[0][1], reverse=True)

                # Determine best ordering of focus region inclusion
                zig_ends_flat = [p for z in zigs for p in (z[0], z[-1])]
                zigs_aug = [list(z) for z in zigs]
                for s in straights:
                    closest_end = self.find_closest_sect(s, zig_ends_flat, return_idx=True)
                    zig, at_end = closest_end // 2, closest_end % 2 == 1
                    if not at_end:
                        zigs_aug[zig] = s + zigs_aug[zig]
                    else:
                        zigs_aug[zig] = zigs_aug[zig] + s

                # Formulate final line
                main_zigorzag_line = [p for z in zigs_aug for p in z]
            elif len(straights) > 0:
                main_zigorzag_line = straights

            zigzag_typewriter_points_final.extend(main_zigorzag_line)

        # Determine best path for typewriter
        section_path = []
        for t in range(len(zigzag_typewriter_points_final) - 1):
            section_path.extend(nxex.shortest_path(self.maze_sections.path_graph,
                                                 zigzag_typewriter_points_final[t], zigzag_typewriter_points_final[t + 1],
                                                 weight='weight')[:-1])
        section_path.append(zigzag_typewriter_points_final[-1])

        # Trace path from tracker path
        nodes_path = self.set_node_path_from_sec_path(section_path)
        raw_path_coords = [n.point for n in nodes_path]
        return raw_path_coords

    def find_trace_section_tour_back_forth(self):
        # Find typewriter set points
        outer_path = self.all_contours_objects[0].section_tracker
        outer_sects = list(set([t.section.coords_sec for t in outer_path]))
        typewriter_points = []
        line_height_sects = self.options.maze_sections_across // (self.options.back_forth_lines - 1)
        back = False
        for l in range(self.options.back_forth_lines):
            # Find line start & end points
            left_sect = self.find_closest_sect((l * line_height_sects, 0), outer_sects)
            right_sect = self.find_closest_sect((l * line_height_sects, self.options.maze_sections_across - 1),
                                                outer_sects)
            if not back:
                typewriter_points.extend([left_sect, right_sect])
            else:
                typewriter_points.extend([right_sect, left_sect])

            back = ~back

        # Determine focus section inclusion
        main_forthbacks_straights = [[] for _ in range(2*self.options.back_forth_lines)]
        main_forthbacks_subforthbacks = [[] for _ in range(2*self.options.back_forth_lines)]
        for focus_region in self.maze_sections.focus_region_sections:
            if len(focus_region) == 0: continue
            sections_coords = np.atleast_2d([s.coords_sec for s in focus_region])

            # Determine method of traversal
            min_y, max_y = np.min(sections_coords[:, 0]), np.max(sections_coords[:, 0])
            min_x, max_x = np.min(sections_coords[:, 1]), np.max(sections_coords[:, 1])
            reg_height, reg_width = max_y - min_y, max_x - min_x

            # Find closest forth or back
            mid_height = (max_y + min_y) // 2
            forth_or_back = mid_height // (line_height_sects//2)
            is_forth, is_back = forth_or_back % 2 == 0, forth_or_back % 2 == 1

            section_coords_ls = [tuple(s) for s in sections_coords.tolist()]

            if reg_height / line_height_sects < self.options.back_forth_traverse_threshold:
                # Traverse in line with typewriting,
                if is_forth:
                    right_insert, left_insert = (max_y, max_x), (min_y, min_x)
                    main_forthbacks_straights[forth_or_back].extend([left_insert, right_insert])
                else:
                    right_insert, left_insert = (self.find_closest_sect((min_y, max_x), section_coords_ls),
                                                 self.find_closest_sect((max_y, min_x), section_coords_ls))
                    main_forthbacks_straights[forth_or_back].extend([right_insert, left_insert])
            else:
                # forthback within carriage return
                subforthback = []
                if is_forth:
                    left_1, right_1  = (min_y, min_x), self.find_closest_sect((min_y, max_x), section_coords_ls)
                    left_2, right_2  = self.find_closest_sect((max_y, min_x), section_coords_ls), (max_y, max_x)
                    subforthback = (left_1, right_1, left_2, right_2)
                else:
                    right_1, left_1 = self.find_closest_sect((min_y, max_x), section_coords_ls), (min_y, min_x)
                    right_2, left_2 = (max_y, max_x), self.find_closest_sect((max_y, min_x), section_coords_ls)
                    subforthback = (right_1, left_1, right_2, left_2)

                # NOTE: append since need to preserve order
                main_forthbacks_subforthbacks[forth_or_back].append(subforthback)

        # Build focus regions into typewriter points
        back_forth_points_final = []
        for l in range(2*self.options.back_forth_lines):
            back_forth_points_final.extend(typewriter_points[l:l + 1])
            straights, forths = main_forthbacks_straights[l], main_forthbacks_subforthbacks[l]
            main_forthorback_line = []

            # Sort by x of leftmost desc
            if len(straights) > 0: straights.sort(key=lambda p: p[1], reverse=True)
            if len(forths) > 0:
                forths.sort(key=lambda z: z[0][1], reverse=True)

                # Determine best ordering of focus region inclusion
                forth_ends_flat = [p for z in forths for p in (z[0], z[-1])]
                forths_aug = [list(z) for z in forths]
                for s in straights:
                    closest_end = self.find_closest_sect(s, forth_ends_flat, return_idx=True)
                    forth, at_end = closest_end // 2, closest_end % 2 == 1
                    if not at_end:
                        forths_aug[forth] = s + forths_aug[forth]
                    else:
                        forths_aug[forth] = forths_aug[forth] + s

                # Formulate final line
                main_forthorback_line = [p for z in forths_aug for p in z]
            elif len(straights) > 0:
                main_forthorback_line = straights

            back_forth_points_final.extend(main_forthorback_line)

        # Determine best path for typewriter
        section_path = []
        for t in range(len(back_forth_points_final) - 1):
            section_path.extend(nxex.shortest_path(self.maze_sections.path_graph,
                                                 back_forth_points_final[t], back_forth_points_final[t + 1],
                                                 weight='weight')[:-1])
        section_path.append(back_forth_points_final[-1])

        # Trace path from tracker path
        nodes_path = self.set_node_path_from_sec_path(section_path)
        raw_path_coords = [n.point for n in nodes_path]
        return raw_path_coords

    def find_trace_section_tour_vertical_zigzag(self):
        # Find typewriter set points
        outer_path = self.all_contours_objects[0].section_tracker
        outer_sects = list(set([t.section.coords_sec for t in outer_path]))
        typewriter_points = []
        line_width_sects = self.options.maze_sections_across // (self.options.vertical_zigzag_lines - 1)
        for l in range(self.options.vertical_zigzag_lines):
            # Find line start & end points
            top_sect = self.find_closest_sect((0, l * line_width_sects), outer_sects)
            bottom_sect = self.find_closest_sect((self.options.maze_sections_across - 1,
                                                 l * line_width_sects + line_width_sects//2),
                                                outer_sects)
            typewriter_points.extend([top_sect, bottom_sect])

        # Determine focus section inclusion
        main_zigzags_straights = [[] for _ in range(2*self.options.vertical_zigzag_lines)]
        main_zigzags_subzigzags = [[] for _ in range(2*self.options.vertical_zigzag_lines)]
        for focus_region in self.maze_sections.focus_region_sections:
            if len(focus_region) == 0: continue
            sections_coords = np.atleast_2d([s.coords_sec for s in focus_region])

            # Determine method of traversal
            min_y, max_y = np.min(sections_coords[:, 0]), np.max(sections_coords[:, 0])
            min_x, max_x = np.min(sections_coords[:, 1]), np.max(sections_coords[:, 1])
            reg_height, reg_width = max_y - min_y, max_x - min_x

            # Find closest zig or zag
            mid_width = (max_x + min_x) // 2
            zig_or_zag = mid_width // (line_width_sects//2)
            is_zig, is_zag = zig_or_zag % 2 == 0, zig_or_zag % 2 == 1

            section_coords_ls = [tuple(s) for s in sections_coords.tolist()]

            if reg_width / line_width_sects < self.options.vertical_zigzag_traverse_threshold:
                # Traverse in line with typewriting,
                if is_zig:
                    bottom_insert, top_insert = (max_y, max_x), (min_y, min_x)
                    main_zigzags_straights[zig_or_zag].extend([top_insert, bottom_insert])
                else:
                    bottom_insert, top_insert  = (self.find_closest_sect((max_y, min_x), section_coords_ls),
                                                 self.find_closest_sect((min_y, max_x), section_coords_ls))
                    main_zigzags_straights[zig_or_zag].extend([bottom_insert, top_insert])
            else:
                # Zigzag within carriage return
                subzigzag = []
                if is_zig:
                    top_1, bottom_1  = (min_y, min_x), self.find_closest_sect((max_y, min_x), section_coords_ls)
                    top_2, bottom_2  = self.find_closest_sect((min_y, max_x), section_coords_ls), (max_y, max_x)
                    subzigzag = (top_1, bottom_1, top_2, bottom_2)
                else:
                    bottom_1, top_1 = self.find_closest_sect((max_y, min_x), section_coords_ls), (min_y, min_x)
                    bottom_2, top_2 = (max_y, max_x), self.find_closest_sect((min_y, max_x), section_coords_ls)
                    subzigzag = (bottom_1, top_1, bottom_2, top_2)

                # NOTE: append since need to preserve order
                main_zigzags_subzigzags[zig_or_zag].append(subzigzag)

        # Build focus regions into typewriter points
        vertical_zigzag_points_final = []
        for l in range(2*self.options.vertical_zigzag_lines):
            vertical_zigzag_points_final.extend(typewriter_points[l:l + 1])
            straights, zigs = main_zigzags_straights[l], main_zigzags_subzigzags[l]
            main_zigorzag_line = []

            # Sort by x of leftmost desc
            if len(straights) > 0: straights.sort(key=lambda p: p[1], reverse=True)
            if len(zigs) > 0:
                zigs.sort(key=lambda z: z[0][1], reverse=True)

                # Determine best ordering of focus region inclusion
                zig_ends_flat = [p for z in zigs for p in (z[0], z[-1])]
                zigs_aug = [list(z) for z in zigs]
                for s in straights:
                    closest_end = self.find_closest_sect(s, zig_ends_flat, return_idx=True)
                    zig, at_end = closest_end // 2, closest_end % 2 == 1
                    if not at_end:
                        zigs_aug[zig] = s + zigs_aug[zig]
                    else:
                        zigs_aug[zig] = zigs_aug[zig] + s

                # Formulate final line
                main_zigorzag_line = [p for z in zigs_aug for p in z]
            elif len(straights) > 0:
                main_zigorzag_line = straights

            vertical_zigzag_points_final.extend(main_zigorzag_line)

        # Determine best path for typewriter
        section_path = []
        for t in range(len(vertical_zigzag_points_final) - 1):
            section_path.extend(nxex.shortest_path(self.maze_sections.path_graph,
                                                 vertical_zigzag_points_final[t], vertical_zigzag_points_final[t + 1],
                                                 weight='weight')[:-1])
        section_path.append(vertical_zigzag_points_final[-1])

        # Trace path from tracker path
        nodes_path = self.set_node_path_from_sec_path(section_path)
        raw_path_coords = [n.point for n in nodes_path]
        return raw_path_coords

    def set_node_path_from_sec_path(self, sections_nodes_path):
        nodes_path = []
        jump_path = False
        prev_sect, cur_sect, next_sect = None, None, None
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

            if len(cur_sect) == 4:
                if inflection_out is not None:
                    cur_tracker_idx = inflection_out['tracker_path_idx']
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
                elif next_sect is not None:
                    #Entering into the abyss, look for next insersection
                    next_trackerized_sect, sections_path_idx = None, -1
                    for j in range(i + 2, len(sections_nodes_path)):
                        if len(sections_nodes_path[j]) == 4:
                            next_trackerized_sect = sections_nodes_path[j]
                            sections_path_idx = j
                            break

                    if next_trackerized_sect is not None:
                        if next_trackerized_sect != cur_sect:
                            #Find best intersection point
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
        return nodes_path

    def find_maximal_polygon_optimized(self, coords, start1, start2, faces):
        coords_s = set([tuple(row) for row in coords])
        coords_s.remove(start1)
        coords_s.remove(start2)
        coords_array = np.array(list(coords_s))

        hull = ConvexHull(coords_array)
        hull_points = coords_array[hull.vertices]

        max_area = 0
        largest_polygon_nd = None
        start1_nd, start2_nd = np.array(start1), np.array(start2)
        #Max out if convex hull smaller than req # faces
        eff_faces = min(faces, hull_points.shape[0] + 2)

        for combination in itertools.combinations(hull_points, eff_faces - 2):
            polygon_points = list(combination) + [start1_nd, start2_nd]
            for poly_perm in itertools.permutations(polygon_points):
                poly_perm_nd = np.array(poly_perm)
                # check that the start points are next to one another.
                matches = np.all(poly_perm_nd == start1_nd, axis=1)
                start1_index = np.where(matches)[0]
                matches = np.all(poly_perm_nd == start2_nd, axis=1)
                start2_index = np.where(matches)[0]

                if abs(start1_index - start2_index) == 1 or abs(start1_index - start2_index) == eff_faces - 1:
                    area = self.calculate_polygon_area(poly_perm)
                    if area > max_area:
                        max_area = area
                        largest_polygon_nd = poly_perm_nd

        if largest_polygon_nd is None:
            return None

        # Reorder the largest polygon to start from start1 and end with start2
        matches = np.all(largest_polygon_nd == start1_nd, axis=1)
        start1_index = np.where(matches)[0][0]
        matches = np.all(largest_polygon_nd == start2_nd, axis=1)
        start2_index = np.where(matches)[0][0]
        largest_polygon = largest_polygon_nd.tolist()

        #Flip around, we need start2 in front to bring around to end of tour
        if (start1_index + 1)%eff_faces == start2_index:
            largest_polygon = list(reversed(largest_polygon))
            start1_index = eff_faces - 1 - start1_index
            start2_index = eff_faces - 1 - start2_index

        ordered_polygon = largest_polygon[start1_index:] + largest_polygon[:start1_index]

        return [tuple(p) for p in ordered_polygon]

    def calculate_polygon_area(self, points):
        """Calculates the area of a polygon using the shoelace formula."""
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.sum(x[:-1] * y[1:] - y[:-1] * x[1:]) + x[-1] * y[0] - y[-1] * x[0])

    def find_inflection_points(self, path_1_seg, path_2_seg, retrieve_idxs=False):
        points_1, points_2 = np.array([n.point for n in path_1_seg]), np.array([n.point for n in path_2_seg])

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

    def find_focus_region_incision_point(self, outer_sections, focus_sections):
        """
            Finds pairings of coords where an outer_sections coord is within a distance
            threshold from a focus_sections coord, and finds the closest two pairings.

            Args:
                outer_sections: NumPy array of outer coordinates (n x 2).
                focus_sections: NumPy array of focus coordinates (m x 2).

            Returns:
                A tuple: (list of pairings, closest_pair_indices)
                    pairings: List of tuples (outer_index, focus_coord).
                    closest_pair_indices: Tuple of two outer indices.
            """

        # Calculate all pairwise distances using broadcasting
        # outer_distinct = np.unique(outer_sections, axis=0)
        distances = np.linalg.norm(outer_sections[:, None, :] - focus_sections[None, :, :], axis=2)

        # Find pairings within the distance threshold
        outer_indices, focus_indices = np.where(distances < self.options.snake_trace_max_jump_from_outer)
        if outer_indices.size == 0:
            return None, None
        #TODO: Figure out some way sort by asc distance first

        #Find closest indexes that are not the same
        outer_disps = np.abs(outer_indices[:, None] - outer_indices[None, :])
        outer_disps[outer_disps == 0] = 9999
        focus_simult = focus_indices[:, None] == focus_indices[None, :]
        outer_disps[focus_simult] = 9999
        closest_indices = np.unravel_index(np.argmin(outer_disps), outer_disps.shape)

        #Return pairings
        if outer_indices[closest_indices[0]] < outer_indices[closest_indices[1]]:
            closest_outer = (outer_indices[closest_indices[0]], outer_indices[closest_indices[1]])
            closest_focus = (focus_indices[closest_indices[0]], focus_indices[closest_indices[1]])
        else:
            closest_outer = (outer_indices[closest_indices[1]], outer_indices[closest_indices[0]])
            closest_focus = (focus_indices[closest_indices[1]], focus_indices[closest_indices[0]])

        return closest_outer, closest_focus

    def find_cluster_max_start_node(self):
        # Convolve with ones to find tightest cluster
        kernel = np.ones((self.options.cluster_start_point_size, self.options.cluster_start_point_size), dtype=np.uint8)
        convolved = convolve2d(self.all_edges_bool.astype(np.uint8), kernel, mode='same')
        max_index = np.argmax(convolved)
        cluster_point = np.unravel_index(max_index, self.all_edges_bool.shape)
        nearest_outer_point = self.find_closest_outer_edge_point(cluster_point)
        start_node = self.find_closest_node_to_edge_point(nearest_outer_point)
        return start_node, start_node.point, start_node.section

    def find_cluster_max_end_node(self, start_node: EdgeNode):
        #Find a point approx across
        start_point = start_node.point
        approx_end_point = ((self.dims[0] - 1) - start_point[0], (self.dims[1] - 1) - start_point[1])
        nearest_outer_point = self.find_closest_outer_edge_point(approx_end_point)
        end_node = self.find_closest_node_to_edge_point(nearest_outer_point)
        return end_node, end_node.point, end_node.section

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








