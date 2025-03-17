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
        # pars.add_argument("--keeporiginal", type=inkex.Boolean, default=False, help="Keep original image on canvas")
        # pars.add_argument("--ltres", type=float, default=1.0, help="Error treshold straight lines")
        # pars.add_argument("--qtres", type=float, default=1.0, help="Error treshold quadratic splines")
        # pars.add_argument("--pathomit", type=int, default=8, help="Noise reduction - discard edge node paths shorter than")
        # pars.add_argument("--rightangleenhance", type=inkex.Boolean, default=True, help="Enhance right angle corners")
        # pars.add_argument("--colorsampling", default="2",help="Color sampling")
        # pars.add_argument("--numberofcolors", type=int, default=16, help="Number of colors to use on palette")
        # pars.add_argument("--mincolorratio", type=int, default=0, help="Color randomization ratio")
        # pars.add_argument("--colorquantcycles", type=int, default=3, help="Color quantization will be repeated this many times")
        # pars.add_argument("--layering", default="0",help="Layering")
        # pars.add_argument("--strokewidth", type=float, default=1.0, help="SVG stroke-width")
        # pars.add_argument("--linefilter", type=inkex.Boolean, default=False, help="Noise reduction line filter")
        # #pars.add_argument("--scale", type=float, default=1.0, help="Coordinate scale factor")
        # pars.add_argument("--roundcoords", type=int, default=1, help="Decimal places for rounding")
        # pars.add_argument("--viewbox", type=inkex.Boolean, default=False, help="Enable or disable SVG viewBox")
        # pars.add_argument("--desc", type=inkex.Boolean, default=False, help="SVG descriptions")
        # pars.add_argument("--blurradius", type=int, default=0, help="Selective Gaussian blur preprocessing")
        # pars.add_argument("--blurdelta", type=float, default=20.0, help="RGBA delta treshold for selective Gaussian blur preprocessing")
        #
        pars.add_argument("--slic_regions", type=int, default=12, help="Number of SLIC regions")
        pars.add_argument("--maze_sections_across", type=int, default=70, help="Gridding density for approx path formation")
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

        pars.add_argument("--trace_technique", type=int, default=2,
                          help="TEMP!! Trace technique")

        pars.add_argument("--snake_trace_max_jump_from_outer", type=int, default=2,
                          help="Maximum jump from outer contour for snake trace")
        pars.add_argument("--snake_details_polygon_faces", type=int, default=7,
                          help="Number of polygon faces for snake details")
        pars.add_argument("--typewriter_lines", type=int, default=5, help="Number of typewriter lines")
        pars.add_argument("--typewriter_traverse_threshold", type=float, default=0.5,
                          help="Traverse threshold for typewriter lines")
        pars.add_argument("--zigzag_typewriter_lines", type=int, default=5, help="Number of zigzag typewriter lines")
        pars.add_argument("--zigzag_typewriter_traverse_threshold", type=float, default=0.3,
                          help="Traverse threshold for zigzag typewriter lines")
        pars.add_argument("--vertical_zigzag_lines", type=int, default=5, help="Number of vertical zigzag lines")
        pars.add_argument("--vertical_zigzag_traverse_threshold", type=float, default=0.3,
                          help="Traverse threshold for vertical zigzag lines")
        pars.add_argument("--back_forth_lines", type=int, default=12, help="Number of back and forth lines")
        pars.add_argument("--back_forth_traverse_threshold", type=float, default=0.3,
                          help="Traverse threshold for back and forth lines")
        pars.add_argument("--simplify_tolerance", type=float, default=0.7, help="Simplify tolerance")
        pars.add_argument("--preview", type=inkex.Boolean, default=True, help="Preview before committing")


    def check_level_to_update(self, data_ret:dataret.DataRetrieval):
        # Collect parameter names and values into a list of dictionaries
        param_data = []
        for param_name, param_value in vars(self.options).items():
            param_data.append({'param_name': param_name, 'param_val': param_value})

        # Create a pandas DataFrame from the list of dictionaries
        params_df = pd.DataFrame(param_data)
        return data_ret.level_of_update(params_df)

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



    # def retrieve_image(self, image):
    #     self.path = self.checkImagePath(image)  # This also ensures the file exists
    #     if self.path is None:  # check if image is embedded or linked
    #         image_string = image.get('{http://www.w3.org/1999/xlink}href')
    #         # find comma position
    #         i = 0
    #         while i < 40:
    #             if image_string[i] == ',':
    #                 break
    #             i = i + 1
    #         img = Image.open(BytesIO(base64.b64decode(image_string[i + 1:len(image_string)])))
    #     else:
    #         img = Image.open(self.path)
    #
    #     # Write the embedded or linked image to temporary directory
    #     if os.name == "nt":
    #         exportfile = "helpers.png"
    #     else:
    #         exportfile = "/tmp/helpers.png"
    #
    #     # if img.mode != 'RGB':
    #     #     img = img.convert('RGB')
    #     # img.save(exportfile, "png")
    #
    #     return img

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

    # def fit_svg(self, exportfile, image):
    #     # Delete the temporary png file again because we do not need it anymore
    #     if os.path.exists(exportfile):
    #         os.remove(exportfile)
    #
    #     # new parse the SVG file and insert it as new group into the current document tree
    #     doc = etree.parse(exportfile + ".svg").getroot()
    #     newGroup = self.document.getroot().add(inkex.Group())
    #     trace_width = None
    #     trace_height = None
    #     if doc.get('width') is not None and doc.get('height') is not None:
    #         trace_width = doc.get('width')
    #         trace_height = doc.get('height')
    #     else:
    #         viewBox = doc.get('viewBox')  # eg "0 0 700 600"
    #         trace_width = viewBox.split(' ')[2]
    #         trace_height = viewBox.split(' ')[3]
    #
    #     # add transformation to fit previous XY coordinates and width/height
    #     # image might also be influenced by other transformations from parent:
    #     parent = image.getparent()
    #     if parent is not None and parent != self.document.getroot():
    #         tpc = parent.composed_transform()
    #         x_offset = tpc.e
    #         y_offset = tpc.f
    #     else:
    #         x_offset = 0.0
    #         y_offset = 0.0
    #     img_w = image.get('width')
    #     img_h = image.get('height')
    #     img_x = image.get('x')
    #     img_y = image.get('y')
    #     if img_w is not None and img_h is not None and img_x is not None and img_y is not None:
    #         # if width/height are not unitless but end with px, mm, in etc. we have to convert to a float number
    #         if img_w[-1].isdigit() is False:
    #             img_w = self.svg.uutounit(img_w)
    #         if img_h[-1].isdigit() is False:
    #             img_h = self.svg.uutounit(img_h)
    #
    #         transform = "matrix({:1.6f}, 0, 0, {:1.6f}, {:1.6f}, {:1.6f})" \
    #             .format(float(img_w) / float(trace_width), float(img_h) / float(trace_height), float(img_x) + x_offset,
    #                     float(img_y) + y_offset)
    #         newGroup.attrib['transform'] = transform
    #     else:
    #         t = image.composed_transform()
    #         img_w = t.a
    #         img_h = t.d
    #         img_x = t.e
    #         img_y = t.f
    #         transform = "matrix({:1.6f}, 0, 0, {:1.6f}, {:1.6f}, {:1.6f})" \
    #             .format(float(img_w) / float(trace_width), float(img_h) / float(trace_height), float(img_x) + x_offset,
    #                     float(img_y) + y_offset)
    #         newGroup.attrib['transform'] = transform
    #
    #     for child in doc.getchildren():
    #         newGroup.append(child)
    #
    #     # Delete the temporary svg file
    #     if os.path.exists(exportfile + ".svg"):
    #         os.remove(exportfile + ".svg")
    #
    #     return newGroup

    def det_img_and_focus_specs(self, image_path, detail_bounds):
        img_focus_specs = []
        img_focus_specs.append(image_path)
        for bounds in detail_bounds:
            # Check if ellipse or rect
            if bounds.tag == inkex.addNS('rect', 'svg'):
                # Get rectangle properties
                img_focus_specs.append(float(bounds.attrib.get('x', 0)))
                img_focus_specs.append(float(bounds.attrib.get('y', 0)))
                img_focus_specs.append(float(bounds.attrib.get('width', 0)))
                img_focus_specs.append(float(bounds.attrib.get('height', 0)))
            elif bounds.tag == inkex.addNS('ellipse', 'svg'):
                # Get ellipse properties
                # Get rectangle properties
                img_focus_specs.append(float(bounds.attrib.get('cx', 0)))
                img_focus_specs.append(float(bounds.attrib.get('cy', 0)))
                img_focus_specs.append(float(bounds.attrib.get('rx', 0)))
                img_focus_specs.append(float(bounds.attrib.get('ry', 0)))
            else:
                raise inkex.AbortExtension("Only ellipses and rectangles are supported as bounds.")

        selection_info_df = pd.DataFrame(img_focus_specs, columns=['selection_info'])

        # Add the 'line' column from the index
        # selection_info_df['line'] = selection_info_df.index

        # Reset the index if you want 'line' to be the first column
        # selection_info_df = selection_info_df[['line', 'selection_info']]  # reorders columns

        return selection_info_df

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

    # def isolate_sub_images(self, detail_bounds, export_file, image_in_svg):
    #     #Determine image offsets
    #     (x_offset, y_offset) = self.get_image_offsets(image_in_svg)
    #
    #     with Image.open(export_file) as image:
    #         image_size = image.size
    #
    #         x_scale = image_size[0]/float(image_in_svg.get('width'))
    #         y_scale = image_size[1]/float(image_in_svg.get('height'))
    #
    #         sub_image_counter = 0
    #         bounds_dicts = []
    #
    #         for bounds in detail_bounds:
    #
    #             # Check if ellipse or rect
    #             if bounds.tag == inkex.addNS('rect', 'svg'):
    #                 # Get rectangle properties
    #                 #TODO: figure out way to crop by float
    #                 x = int((float(bounds.attrib.get('x', 0)) - x_offset)*x_scale)
    #                 y = int((float(bounds.attrib.get('y', 0)) - y_offset)*y_scale)
    #                 width = int(float(bounds.attrib.get('width', 0))*x_scale)
    #                 height = int(float(bounds.attrib.get('height', 0))*y_scale)
    #                 local_origin = (x, y)
    #
    #                 # Crop the image to the rectangle
    #                 bbox = (x, y, x + width, y + height)
    #
    #                 cropped_image = image
    #
    #             elif bounds.tag == inkex.addNS('ellipse', 'svg'):
    #                 # Get ellipse properties
    #                 cx = int((float(bounds.attrib.get('cx', 0)) - x_offset)*x_scale)
    #                 cy = int((float(bounds.attrib.get('cy', 0)) - y_offset)*y_scale)
    #                 rx = int(float(bounds.attrib.get('rx', 0))*x_scale)
    #                 ry = int(float(bounds.attrib.get('ry', 0))*y_scale)
    #                 local_origin = np.array([cx - rx, cy - ry])
    #
    #                 # Create a mask for the ellipse
    #                 mask = Image.new('L', image.size, 0)
    #                 draw = ImageDraw.Draw(mask)
    #                 bbox = (cx - rx, cy - ry,
    #                         cx + rx, cy + ry)
    #                 draw.ellipse(bbox, fill=255)
    #
    #                 # Apply the mask to the image
    #                 cropped_image = Image.new('RGBA', image.size)
    #                 self.msg(str(image.size[0]))
    #                 cropped_image.paste(image, (0, 0), mask)
    #             else:
    #                 raise inkex.AbortExtension("Only ellipses and rectangles are supported as bounds.")
    #
    #             # Crop the bounding box of the rect or ellipse
    #             cropped_image = cropped_image.crop(bbox)
    #
    #             # Save the resulting image
    #             #TODO: need to save temp sub images?
    #             output_path = "sub_image_" + str(sub_image_counter) + ".png"
    #             sub_image_counter += 1
    #             cropped_image.save(output_path)
    #             inkex.utils.debug(f"Saved cropped image to {output_path}")
    #
    #             #Save to dict
    #             bounds_dicts.append({"localorigin": local_origin, "imageobject": cropped_image,
    #                 "imagepath": output_path})
    #
    #     return bounds_dicts

    def effect(self):
        # internal overwrite for scale:
        self.options.scale = 1.0
        data_ret = dataret.DataRetrieval()

        images, detail_bounds = [], []
        if len(self.svg.selected) == 0: self.svg.selection = self.svg.descendants().filter(*self.select_all)
        images = self.svg.selection.filter(inkex.Image).values()
        detail_bounds= self.svg.selection.filter(inkex.Rectangle, inkex.Ellipse).values()

        # if len(self.svg.selected) > 0:
        #     #Grab selected items if selected
        #     images = self.svg.selection.filter(inkex.Image).values()
        #     detail_bounds= self.svg.selection.filter(inkex.Rectangle, inkex.Ellipse).values()
        # else:
        #     #Else grab all on doc
        #     for child in self.svg.getchildren():
        #         if str(child) == "g":
        #             for sub_child in child:
        #                 if str(sub_child) in ("image"):
        #                     self.svg.selected[sub_child.get('id')] = sub_child
        #                 elif str(sub_child) in ("ellipse", "rect"):
        #                     detail_bounds.append(sub_child)
        #     images = self.svg.selection.filter(inkex.Image).values()
        #     detail_bounds = self.svg.selection.filter(inkex.Rectangle, inkex.Ellipse).values()

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
                img_and_focus_specs_df = self.det_img_and_focus_specs(image_path, detail_bounds)
                update_level = data_ret.get_selection_match_level(img_and_focus_specs_df)

                self.msg("post get selection match update level: " + str(update_level))

                if update_level > 0:
                    #If selection matches, check if params have changed
                    update_level = min(self.check_level_to_update(data_ret), update_level)

                self.msg("update level: " + str(update_level))

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
                        buildscr.build_level_1_scratch(img_cv, normalized_focus_region_specs, self.options, objects)
                        buildscr.build_level_2_scratch(self.options, objects)
                        buildscr.build_level_3_scratch(self.options, objects)
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
                        buildscr.build_level_3_scratch(self.options, objects)
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
                        buildscr.build_level_3_scratch(self.options, objects)
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
                    image_size = (objects['maze_sections'].img_width, objects['maze_sections'].img_height)
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

                    # Remove old preview layers, whenever preview mode is enabled
                    for node in self.svg:
                        if node.tag in ('{http://www.w3.org/2000/svg}g', 'g'):
                            if node.get('{http://www.inkscape.org/namespaces/inkscape}groupmode') == 'layer':
                                layer_name = node.get('{http://www.inkscape.org/namespaces/inkscape}label')
                                if layer_name == 'Preview':
                                    self.svg.remove(node)

                    if self.options.preview:
                        # Create a temporary layer & group for the preview
                        preview_layer = inkex.etree.Element(inkex.addNS('g', 'svg'),
                                                      None, nsmap=inkex.NSS)
                        preview_layer.set(inkex.addNS('groupmode', 'inkscape'), 'layer')
                        preview_layer.set(inkex.addNS('label', 'inkscape'), 'Preview')
                        preview_group = inkex.etree.SubElement(preview_layer, inkex.addNS('g', 'svg'))
                        preview_group.set('id', 'preview_group')  # give the group an id so it can be found later.

                        # Create the path element
                        path_element = inkex.etree.SubElement(preview_group, inkex.addNS('path', 'svg'))
                        path_element.set('d', commands_str)
                        path_element.set('style', 'stroke:red; stroke-width:2; fill:none;')

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