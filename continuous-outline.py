from logging import exception

import inkex
import os
import urllib.parse as urllib_parse
import urllib.request as urllib_request
# from PIL import Image
# from io import BytesIO
# import base64
import numpy as np
import pandas as pd

import helpers.caching.DataRetrieval as dataret

"""
Extension for InkScape 1.X
Features
 - will vectorize your beautiful image into a more beautiful SVG trace with separated infills(break apart into single surfaces like a puzzle)
 
Author: Mario Voigt / FabLab Chemnitz
Mail: mario.voigt@stadtfabrikanten.org
Date: 18.08.2020
Last patch: 24.04.2021
License: GNU GPL v3

Used version of imagetracerjs: https://github.com/jankovicsandras/imagetracerjs/commit/4d0f429efbb936db1a43db80815007a2cb113b34

"""

class continuous_outline(inkex.EffectExtension):
    def checkImagePath(self, element):
        xlink = element.get('xlink:href')
        if xlink and xlink[:5] == 'data:':
            # No need, data already embedded
            return

        url = urllib_parse.urlparse(xlink)
        href = urllib_request.url2pathname(url.path)

        # Primary location always the filename itself.
        path = self.absolute_href(href or '')

        # Backup directory where we can find the image
        if not os.path.isfile(path):
            path = element.get('sodipodi:absref', path)

        if not os.path.isfile(path):
            inkex.errormsg('File not found "{}". Unable to embed image.').format(path)
            return

        if (os.path.isfile(path)):
            return path

    def add_arguments(self, pars):
        pars.add_argument("--tab")
        pars.add_argument("--slic_regions", type=int, default=12, help="Number of SLIC regions")
        pars.add_argument("--transparancy_cutoff", type=float, default=0.1, help="max % transparent considered background")
        pars.add_argument("--maze_sections_across", type=int, default=70, help="Gridding density for approx path formation")
        pars.add_argument("--constrain_slic_within_mask", type=inkex.Boolean, default=False,
                          help="Omit lines outside of mask")
        pars.add_argument("--dumb_node_optional_weight", type=int, default=1, help="Weight for optional dumb nodes")
        pars.add_argument("--dumb_node_optional_max_variable_weight", type=int, default=6,
                          help="Max variable weight for optional dumb nodes")
        pars.add_argument("--dumb_node_min_opt_weight_reduced", type=int, default=1,
                          help="Minimum reduced weight for optional dumb nodes")
        pars.add_argument("--dumb_node_blank_weight", type=int, default=200, help="Weight for blank dumb nodes")
        pars.add_argument("--dumb_node_opt_jump_weight", type=int, default=1,
                          help="Weight for optional dumb node jumps")
        pars.add_argument("--dumb_node_req_jump_weight", type=int, default=1,
                          help="Weight for required dumb node jumps")
        pars.add_argument("--dumb_node_required_weight", type=int, default=1, help="Weight for required dumb nodes")
        pars.add_argument("--max_inner_path_seg_manhatten_length", type=int, default=50,
                          help="Maximum Manhattan length for inner path segments")
        pars.add_argument("--outer_contour_length_cutoff", type=int, default=200,
                          help="Length cutoff for outer contours")
        pars.add_argument("--inner_contour_length_cutoff", type=int, default=20,
                          help="Length cutoff for inner contours")
        pars.add_argument("--inner_contour_variable_weights", type=inkex.Boolean, default=True,
                          help="Enable variable weights for inner contours")
        pars.add_argument("--scorched_earth", type=inkex.Boolean, default=True, help="Enable scorched earth mode")
        pars.add_argument("--scorched_earth_weight_multiplier", type=int, default=6,
                          help="Weight multiplier for scorched earth mode")
        pars.add_argument("--simplify_tolerance", type=float, default=0.7, help="Simplify tolerance")
        pars.add_argument("--preview", type=inkex.Boolean, default=True, help="Preview before committing")


    def check_level_to_update(self, data_ret:dataret.DataRetrieval):
        # Collect parameter names and values into a list of dictionaries
        param_data = []
        for param_name, param_value in vars(self.options).items():
            param_data.append({'param_name': str(param_name), 'param_val': str(param_value)})

        # Create a pandas DataFrame from the list of dictionaries
        params_df = pd.DataFrame(param_data)
        return data_ret.level_of_update(self, params_df)

    def set_params_into_db(self, data_ret:dataret.DataRetrieval):
        # Collect parameter names and values into a list of dictionaries
        param_data = []
        for param_name, param_value in vars(self.options).items():
            param_data.append({'param_name': param_name, 'param_val': param_value})

        # Create a pandas DataFrame from the list of dictionaries, set col type to str
        params_df = pd.DataFrame(param_data)
        params_df = params_df.map(str)

        #Set into db
        data_ret.clear_and_set_single_table("ParamsVals", params_df)

    def get_image_offsets(self, svg_image):
        # Determine image offsets
        parent = svg_image.getparent()
        x, y = svg_image.get('x'), svg_image.get('y')
        if x is None: x = 0
        if y is None: y = 0
        if parent is not None and parent != self.document.getroot():
            tpc = parent.composed_transform()
            x_offset = tpc.e + float(x)
            y_offset = tpc.f + float(y)
        else:
            x_offset = float(x)
            y_offset = float(y)

        return (x_offset, y_offset)

    def det_img_and_focus_specs(self, image_path, detail_bounds, approx_trace_path_string):
        img_focus_specs = []
        img_focus_specs.append(str(image_path))
        for bounds in detail_bounds:
            # Check if ellipse or rect
            if bounds.tag == inkex.addNS('rect', 'svg'):
                # Get rectangle properties
                img_focus_specs.append(str(float(bounds.attrib.get('x', 0))))
                img_focus_specs.append(str(float(bounds.attrib.get('y', 0))))
                img_focus_specs.append(str(float(bounds.attrib.get('width', 0))))
                img_focus_specs.append(str(float(bounds.attrib.get('height', 0))))
            elif bounds.tag == inkex.addNS('ellipse', 'svg'):
                # Get ellipse properties
                # Get rectangle properties
                img_focus_specs.append(str(float(bounds.attrib.get('cx', 0))))
                img_focus_specs.append(str(float(bounds.attrib.get('cy', 0))))
                img_focus_specs.append(str(float(bounds.attrib.get('rx', 0))))
                img_focus_specs.append(str(float(bounds.attrib.get('ry', 0))))
            else:
                raise inkex.AbortExtension("Only ellipses and rectangles are supported as bounds.")

        #TODO: If only this changes level=3 once config setting up modify with ext input screen open
        img_focus_specs.append(str(approx_trace_path_string))

        selection_info_df = pd.DataFrame(img_focus_specs, columns=['selection_info'])

        # Add the 'line' column from the index
        # selection_info_df['line'] = selection_info_df.index

        # Reset the index if you want 'line' to be the first column
        # selection_info_df = selection_info_df[['line', 'selection_info']]  # reorders columns

        return selection_info_df

    def form_approx_control_points_normalized(self, approx_ctrl_points:list, image_in_svg):
        #Convert ctrl points to img yx format
        approx_ctrl_points_nd = np.array([(p[1], p[0]) for p in approx_ctrl_points])

        #Determine image offsets
        offsets_xy = self.get_image_offsets(image_in_svg)
        offsets_nd = np.array((offsets_xy[1], offsets_xy[0]))
        norms = np.array((float(image_in_svg.get('height')), float(image_in_svg.get('width'))))

        #Determined shifted and normalized points
        formed_points_nd = (approx_ctrl_points_nd - offsets_nd)/norms
        return formed_points_nd


    def form_focus_region_specs_normalized(self, detail_bounds, image_in_svg):
        #Determine image offsets
        (x_offset, y_offset) = self.get_image_offsets(image_in_svg)
        (x_norm, y_norm) = float(image_in_svg.get('width')),  float(image_in_svg.get('height'))
        bounds_out = []

        #Determined shifted and normalized bounds
        for bounds in detail_bounds:
            # Check if ellipse or rect
            if bounds.tag == inkex.addNS('rect', 'svg'):
                # Get rectangle properties
                x = (float(bounds.attrib.get('x', 0)) - x_offset) / x_norm
                y = (float(bounds.attrib.get('y', 0)) - y_offset) / y_norm
                width = float(bounds.attrib.get('width', 0)) / x_norm
                height = float(bounds.attrib.get('height', 0)) / y_norm
                bounds_out.append({'form': "rectangle", 'x': x, 'y': y, 'width': width, 'height': height})
            elif bounds.tag == inkex.addNS('ellipse', 'svg'):
                # Get ellipse properties
                cx = (float(bounds.attrib.get('cx', 0)) - x_offset) / x_norm
                cy = (float(bounds.attrib.get('cy', 0)) - y_offset) / y_norm
                rx = float(bounds.attrib.get('rx', 0)) / x_norm
                ry = float(bounds.attrib.get('ry', 0)) / y_norm
                bounds_out.append({'form': "ellipse", 'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry})
            else:
                raise inkex.AbortExtension("Only ellipses and rectangles are supported as bounds.")

        return bounds_out

    def get_straight_line_points(self, path_string):
        """Extracts points from a straight line path string without using re."""
        #TODO: Make this less hokey!!
        parts = path_string.split(" ")
        if not parts:
            self.msg("Fail split")
            return None

        if not parts[0].upper().startswith('M'):
            self.msg("Fail M")
            return None

        try:
            points = [[float(x) for x in parts[1].split(',')]]
            # self.msg(points[0])
            for part in parts[2:]:
                # self.msg(part)
                if part.upper().startswith('L'): continue
                point = [float(x) for x in part.split(',')]
                points.append(point)
                # self.msg(point)

            return points
        except ValueError:
            return None

    def effect(self):
        # internal overwrite for scale:
        self.options.scale = 1.0
        data_ret = dataret.DataRetrieval()

        # Remove old preview layers, whenever preview mode is enabled
        for node in self.svg:
            if node.tag in ('{http://www.w3.org/2000/svg}g', 'g'):
                if node.get('{http://www.inkscape.org/namespaces/inkscape}groupmode') == 'layer':
                    layer_name = node.get('{http://www.inkscape.org/namespaces/inkscape}label')
                    if layer_name == 'Preview':
                        self.svg.remove(node)

        if len(self.svg.selected) == 0:
            self.svg.selection = self.svg.descendants().filter(*self.select_all)
        images = self.svg.selection.filter(inkex.Image).values()
        detail_bounds= self.svg.selection.filter(inkex.Rectangle, inkex.Ellipse).values()

        approx_traces = []
        for id, node in self.svg.selected.items():
            # Check if the node is a path element
            if node.tag == inkex.addNS('path', 'svg'):
                approx_traces.append(node)
        for child in approx_traces:
            self.msg(child)

        match(len(approx_traces)):
            case 0:
                self.msg("Please build an approx trace line")
                data_ret.close_connection()
                return
            case 1:
                pass
            case _:
                self.msg("Please build a single approx trace line, also remove or deselect all other existing paths")
                data_ret.close_connection()
                return

        match len(images):
            case 0:
                if len(self.svg.selected) > 0:
                    self.msg("No images found in selection! Check if you selected a group instead.")
                else:
                    self.msg("No images found in document!")
            case 1:
                svg_image = None
                for img in images:
                    svg_image = img
                    break
                image_path = self.checkImagePath(svg_image)
                img_is_embedded = False
                if image_path is None:  # check if image is embedded or linked
                    image_path = svg_image.get('{http://www.w3.org/1999/xlink}href')
                    img_is_embedded = True

                #Check to verify selection/doc/params haven't changed since last run
                approx_trace = None
                for path in approx_traces:
                    approx_trace = path
                    break
                approx_trace_path_string = approx_trace.get('d')
                img_and_focus_specs_df = self.det_img_and_focus_specs(image_path, detail_bounds,
                                                                      approx_trace_path_string).dropna()
                update_level = data_ret.get_selection_match_level(self, img_and_focus_specs_df)

                self.msg("post-get selection match update level: " + str(update_level))

                if update_level > 0:
                    #If selection matches, check if params have changed
                    update_level = min(self.check_level_to_update(data_ret), update_level)

                self.msg("update level: " + str(update_level))
                #TODO: Get object creation working maybe, might not be worth it???
                if update_level == 3: update_level = 2

                #Get approx control points for final path
                approx_ctrl_points = self.get_straight_line_points(approx_trace_path_string)
                formed_normalized_ctrl_points_nd = self.form_approx_control_points_normalized(approx_ctrl_points,
                                                                                             svg_image)

                self.msg("Constrain: " + str(self.options.constrain_slic_within_mask))

                #Retrieve or calculate data as needed
                objects = {}
                match update_level:
                    case 0 | 1:
                        import helpers.build_objects as buildscr
                        import helpers.caching.set_data_by_level as setdb
                        import cv2

                        #Build up image data
                        if img_is_embedded:
                            import base64
                            # find comma position
                            i = 0
                            while i < 40:
                                if image_path[i] == ',':
                                    break
                                i = i + 1
                            img_data = base64.b64decode(image_path[i + 1:len(image_path)])
                            img_array = np.frombuffer(img_data, dtype=np.uint8)
                            img_cv = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                        else:
                            img_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

                        #Retrieve Level 1 objects from db
                        _, dataframes = data_ret.retrieve_and_wipe_data(0)

                        #Form up normalized focus regions
                        normalized_focus_region_specs = self.form_focus_region_specs_normalized(detail_bounds, svg_image)

                        #Levels 1-4 objects from scratch
                        for region in normalized_focus_region_specs:
                            self.msg(region)
                        buildscr.build_level_1_scratch(self, img_cv, normalized_focus_region_specs, self.options, objects)
                        buildscr.build_level_2_scratch(self.options, objects)
                        buildscr.build_level_3_scratch(self, self.options, objects, formed_normalized_ctrl_points_nd)
                        buildscr.build_level_4_scratch(self.options, objects)
                        # for key in objects.keys(): self.msg(key)

                        #Build Level 1-4 data into dataframes
                        setdb.set_level_1_data(dataframes, objects)
                        setdb.set_level_2_data(dataframes, objects)
                        setdb.set_level_3_data(dataframes, objects)
                        setdb.set_level_4_data(dataframes, objects)

                        #Set data into db
                        data_ret.set_data(dataframes, min_level=0)
                        # self.msg(str(update_level))

                    case 2:
                        import helpers.caching.build_objects_from_db as builddb
                        import helpers.build_objects as buildscr
                        import helpers.caching.set_data_by_level as setdb

                        #Retrieve Level 1 objects from db
                        retrieved, dataframes = data_ret.retrieve_and_wipe_data(1)
                        builddb.build_level_1_data(self.options, retrieved, objects)

                        #Levels 2-4 objects from scratch
                        buildscr.build_level_2_scratch(self.options, objects)
                        buildscr.build_level_3_scratch(self, self.options, objects, formed_normalized_ctrl_points_nd)
                        buildscr.build_level_4_scratch(self.options, objects)

                        #Build Level 2-4 data into dataframes
                        setdb.set_level_2_data(dataframes, objects)
                        setdb.set_level_3_data(dataframes, objects)
                        setdb.set_level_4_data(dataframes, objects)

                        #Set data into db
                        data_ret.set_data(dataframes, min_level=2)
                    case 3:
                        import helpers.caching.build_objects_from_db as builddb
                        import helpers.build_objects as buildscr
                        import helpers.caching.set_data_by_level as setdb

                        #Retrieve Level 1-2 objects from db
                        retrieved, dataframes = data_ret.retrieve_and_wipe_data(2)
                        builddb.build_level_1_data(self.options, retrieved, objects)
                        builddb.build_level_2_data(self.options, retrieved, objects)

                        #Levels 3-4 objects from scratch
                        buildscr.build_level_3_scratch(self, self.options, objects, formed_normalized_ctrl_points_nd)
                        buildscr.build_level_4_scratch(self.options, objects)

                        #Build Level 3-4 data into dataframes
                        setdb.set_level_3_data(dataframes, objects)
                        setdb.set_level_4_data(dataframes, objects)

                        #Set data into db
                        data_ret.set_data(dataframes, min_level=3)
                    case 4:
                        import helpers.caching.build_objects_from_db as builddb
                        import helpers.build_objects as buildscr
                        import helpers.caching.set_data_by_level as setdb

                        #Retrieve img height and width (since cannot retrieve direct from non-existent Sections object)
                        sections_df = data_ret.read_sql_table("Sections", data_ret.conn)
                        objects["img_height"], objects["img_width"] = (sections_df.at[0, 'img_height'],
                                                                       sections_df.at[0, 'img_width'])

                        #Retrieve Level 3 objects from db
                        #NOTE: we dont need earlier since this is just forming up from raw path
                        retrieved, dataframes = data_ret.retrieve_and_wipe_data(3)
                        #TODO: Only retrieve from raw table, wipe formed
                        builddb.build_level_3_data(self.options, retrieved, objects)

                        #Levels 4 objects from scratch
                        buildscr.build_level_4_scratch(self.options, objects)

                        #Build Level 4 data into dataframes
                        setdb.set_level_4_data(dataframes, objects)

                        #Set data into db
                        data_ret.set_data(dataframes, min_level=4)
                    case _:
                        import helpers.caching.build_objects_from_db as builddb

                        #Retrieve Level 4 objects (no setting required)
                        retrieved = {}
                        retrieved['FormedPath'] = data_ret.read_sql_table('FormedPath', data_ret.conn)
                        builddb.build_level_4_data(self.options, retrieved, objects)

                        #Retrieve img height and width (since cannot retrieve direct from non-existent Sections object)
                        sections_df = data_ret.read_sql_table("Sections", data_ret.conn)
                        objects["img_height"], objects["img_width"] = (sections_df.at[0, 'img_height'],
                                                                       sections_df.at[0, 'img_width'])
                attributes = vars(self.options)
                # for key, value in attributes.items():
                #     self.msg(f"{key}: {value}")
                #Format output curve to fit into doc
                #NOTE: Going from y, x to x, y
                if len(objects['formed_path']) == 0:
                    self.msg("No path was possible, please change up settings/bounding rects/ellipses and try again")
                    if self.options.preview:
                        # Set updated params into db
                        self.set_params_into_db(data_ret)
                    else:
                        #Clear the database
                        data_ret.wipe_data()
                else:
                    formed_path_nd = np.array(objects['formed_path'])
                    formed_path_xy = formed_path_nd[:, [1, 0]]

                    #Determine scaling
                    image_size = (objects["img_width"], objects["img_height"])
                    x_scale = image_size[0] / float(svg_image.get('width'))
                    y_scale = image_size[1] / float(svg_image.get('height'))
                    scale_nd = np.array([x_scale, y_scale])

                    # Determine image offsets
                    main_image_offsets = np.array(self.get_image_offsets(svg_image))

                    #Offset main contour to line up with master photo on svg
                    formed_path_shifted = (formed_path_xy/scale_nd + main_image_offsets).tolist()

                    #Build the path commands
                    commands = []
                    for i, point in enumerate(formed_path_shifted):
                        if i == 0:
                            commands.append(['M', point])  # Move to the first point
                        else:
                            commands.append(['L', point])  # Line to the next point
                        # self.msg(str(point))
                    # commands.append(['Z'])  # Close path
                    command_strings = [
                        f"{cmd_type} {x},{y}" for cmd_type, (x, y) in commands
                    ]
                    commands_str = " ".join(command_strings)

                    if self.options.preview:
                        # Create a temporary layer & group for the preview
                        preview_layer = inkex.etree.Element(inkex.addNS('g', 'svg'),
                                                      None, nsmap=inkex.NSS)
                        preview_layer.set(inkex.addNS('groupmode', 'inkscape'), 'layer')
                        preview_layer.set(inkex.addNS('label', 'inkscape'), 'Preview')
                        preview_layer.set(inkex.addNS('lock', 'inkscape'), 'true')
                        preview_layer.set(inkex.addNS('insensitive', 'inkscape'), 'true')

                        preview_group = inkex.etree.SubElement(preview_layer, inkex.addNS('g', 'svg'))
                        preview_group.set('id', 'preview_group')  # give the group an id so it can be found later.
                        preview_group.set(inkex.addNS('lock', 'inkscape'), 'true')  # give the group an id so it can be found later.

                        # Create the path element
                        path_element = inkex.etree.SubElement(preview_group, inkex.addNS('path', 'svg'))
                        path_element.set('d', commands_str)
                        path_element.set('style', 'stroke:red; stroke-width:2; fill:none;')
                        path_element.set(inkex.addNS('lock', 'inkscape'), 'true')

                        self.svg.append(preview_layer)

                        # Set updated params into db
                        self.set_params_into_db(data_ret)
                    else:
                        # Add a new path element to the SVG
                        path_element = inkex.PathElement()
                        path_element.set('d', commands_str)  # Set the path data
                        path_element.style = {'stroke': 'black', 'fill': 'none'}
                        self.svg.get_current_layer().add(path_element)

                        #Clear the database
                        data_ret.wipe_data()

            case _:
                if len(self.svg.selected) > 0:
                    self.msg("Multiple images found in selection! Please select only 1, plus any focal regions desired.")
                else:
                    self.msg("Multiple images found in document, please select only 1, plus any focal regions desired.")

        data_ret.close_connection()

if __name__ == '__main__':
    try:
        import inkscape_ExtensionDevTools

        inkscape_ExtensionDevTools.inkscape_run_debug()
    except:
        pass
    continuous_outline().run()