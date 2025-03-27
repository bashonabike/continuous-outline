import time
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
import re

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

#TODO: Support transformed images?? Maybe find some existing code for this
#TODO: Neural net randomize assortment opt and req sections. Maybe can then do straight line approx randomized slic each edge from one part section to other side, build into random blobs

class continuous_outline(inkex.EffectExtension):
    def checkImagePath(self, element):
        xlink = element.get('xlink:href')
        if xlink and xlink[:5] == 'data:':
            # No need, data already embedded
            return

        url = urllib_parse.urlparse(xlink)
        href = urllib_request.url2pathname(url.path)


        # Primary location always the filename itself.
        path = self.absolute_href(href)
        if os.name == 'nt':
            path_formed = path.replace("\\", "\\\\")
        else:
            path_formed = path

        # Backup directory where we can find the image
        if not os.path.isfile(path_formed):
            path = element.get('sodipodi:absref', path)
        else:
            path = path_formed

        if not os.path.isfile(path):
            inkex.errormsg('File not found "{}". Unable to embed image.').format(path)
            return

        if (os.path.isfile(path)):
            return path

    def add_arguments(self, pars):
        pars.add_argument("--tab")
        pars.add_argument("--mask_only", type=inkex.Boolean, default=False,
                          help="No inner SLIC")
        pars.add_argument("--mask_only_when_complicated_background", type=inkex.Boolean, default=False,
                          help="No inner SLIC/Canny if background of mask is complicated (i.e. lots of holes)")
        pars.add_argument("--attempt_mask_jpeg", type=inkex.Boolean, default=False,
                          help="Attempt to mask outline of any JPEG (use Max transparency % considered background)")
        pars.add_argument("--canny_hull", type=inkex.Boolean, default=False,
                          help="Use Canny instead of SLIC (works best when many sharp edges with minimal colour differences between regions)")



        pars.add_argument("--canny_quantize_bins", type=int, default=5,
                          help="Colour quantization bins for Canny (0 is natural)")
        pars.add_argument("--canny_min_contour_length", type=int, default=200,
                          help="Min edge length for Canny")

        pars.add_argument("--cuda_slic", type=inkex.Boolean, default=True,
                          help="Use CUDA SLIC")
        pars.add_argument("--slic_regions", type=int, default=12, help="Number of SLIC regions")
        pars.add_argument("--slic_max_image_resolution", type=int, default=1000,
                          help="Max resolution in either dimension for SLIC image processing (will downsample if needed)")
        pars.add_argument("--slic_lanczos", type=inkex.Boolean, default=False,
                          help="Use Lanczos interpolation for SLIC downscaling (sharper)")
        pars.add_argument("--slic_snap_edges", type=inkex.Boolean, default=False,
                          help="Snap SLIC to edges")
        pars.add_argument("--slic_lab", type=inkex.Boolean, default=True,
                          help="LAB for SLIC")
        pars.add_argument("--slic_greyscale", type=inkex.Boolean, default=False,
                          help="Greyscale for SLIC")
        pars.add_argument("--slic_multi_image_conjoin_processing", type=inkex.Boolean, default=False,
                          help="Conjoin processing of images")
        pars.add_argument("--slic_multi_image_blend_mode", type=str, dest="slic_multi-image-blend-mode",
                          default="Overlay",
                          choices=["Overlay"],
                          help="Blend mode for multi image conjoined processing")
        pars.add_argument("--mask_retain_inner_transparencies", type=inkex.Boolean, default=False,
                          help="Retain inner transparency contours")
        pars.add_argument("--transparancy_cutoff", type=float, default=0.1, help="max % transparent considered background")
        pars.add_argument("--mask_retain_inner_erosion", type=int, default=0,
                          help="Retaining inner transparency erosion (in pixels)")
        pars.add_argument("--maze_sections_across", type=int, default=70, help="Gridding density for approx path formation")
        pars.add_argument("--constrain_slic_within_mask", type=inkex.Boolean, default=False,
                          help="Omit lines outside of mask")

        pars.add_argument("--max_magnet_lock_dist", type=int, default=100,
                          help="Max distance for magnet locking onto edges")
        pars.add_argument("--prefer_outer_contours_locking", type=inkex.Boolean, default=True,
                          help="Prefer outer contours for locking onto edges")


        pars.add_argument("--dumb_node_optional_weight", type=int, default=1, help="Weight for optional dumb nodes")
        pars.add_argument("--dumb_node_optional_max_variable_weight", type=int, default=6,
                          help="Max variable weight for optional dumb nodes")
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
        pars.add_argument("--inner_contour_length_cutoff", type=int, default=10,
                          help="Length cutoff for inner contours")
        pars.add_argument("--inner_contour_variable_weights", type=inkex.Boolean, default=True,
                          help="Enable variable weights for inner contours")
        pars.add_argument("--trace_inner_too", type=inkex.Boolean, default=False,
                          help="Enable tracing of inner contours (dep on rough trace input)")
        pars.add_argument("--scorched_earth", type=inkex.Boolean, default=True, help="Enable scorched earth mode")
        pars.add_argument("--scorched_earth_weight_multiplier", type=int, default=2,
                          help="Weight multiplier for scorched earth mode")
        pars.add_argument("--simplify_intelligent_straighting_cutoff", type=int, default=20,
                          help="Preserve desired details but simplify long sections greater than length")
        pars.add_argument("--simplify_tolerance", type=float, default=0.7,
                          help="Simplify tolerance (lower is sharper)")
        pars.add_argument("--simplify_preserve_topology", type=inkex.Boolean, default=True,
                          help="Preserve topology on simplify")
        pars.add_argument("--blip_max_thickness", type=int, default=0,
                          help="Max thickness of blip for removal")
        pars.add_argument("--blip_max_perimeter", type=int, default=0,
                          help="Max perimeter of blip for removal")
        pars.add_argument("--blip_acuteness_threshold", type=float, default=0.15,
                          help="Acuteness threshold for blip removal (lower is sharper)")
        pars.add_argument("--dither", type=int, default=0,
                          help="Add periodic dither to line for texture")

        #TODO: More sophisticated jumping, so don't need to simplify in inkscape which washes out desired features (goal!)


        pars.add_argument("--preview", type=inkex.Boolean, default=True, help="Preview before committing")


    def _tokenize_path(self, pathdef,COMMAND_RE,  COMMANDS, FLOAT_RE):
        for x in COMMAND_RE.split(pathdef):
            if x in COMMANDS:
                yield x
            for token in FLOAT_RE.findall(x):
                yield token

    def _parse_path(self, pathdef, current_pos=0j, tree_element=None):

        COMMANDS = set('MmZzLlHhVvCcSsQqTtAa')
        UPPERCASE = set('MZLHVCSQTA')

        COMMAND_RE = re.compile(r"([MmZzLlHhVvCcSsQqTtAa])")
        FLOAT_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
        #PULLED FROM SVGPATHTOOLS (thanks!!)
        # In the SVG specs, initial movetos are absolute, even if
        # specified as 'm'. This is the default behavior here as well.
        # But if you pass in a current_pos variable, the initial moveto
        # will be relative to that current_pos. This is useful.
        elements = list(self._tokenize_path(pathdef,COMMAND_RE,  COMMANDS, FLOAT_RE))
        # Reverse for easy use of .pop()
        elements.reverse()
        start_pos = None
        command = None
        coord_points = []
        absolute = False

        while elements:

            if elements[-1] in COMMANDS:
                # New command.
                last_command = command  # Used by S and T
                command = elements.pop()
                absolute = command in UPPERCASE
                command = command.upper()
            else:
                # If this element starts with numbers, it is an implicit command
                # and we don't change the command. Check that it's allowed:
                if command is None:
                    raise ValueError("Unallowed implicit command in %s, position %s" % (
                        pathdef, len(pathdef.split()) - len(elements)))
                last_command = command  # Used by S and T

            if command == 'M':
                # Moveto command.
                x = elements.pop()
                y = elements.pop()
                coord_points.append((float(x), float(y)))
                pos = float(x) + float(y) * 1j
                if absolute:
                    current_pos = pos
                else:
                    current_pos += pos

                # when M is called, reset start_pos
                # This behavior of Z is defined in svg spec:
                # http://www.w3.org/TR/SVG/paths.html#PathDataClosePathCommand
                start_pos = current_pos

                # Implicit moveto commands are treated as lineto commands.
                # So we set command to lineto here, in case there are
                # further implicit commands after this moveto.
                command = 'L'

            elif command == 'Z':
                # Close path
                if not (current_pos == start_pos):
                    coord_points.append((start_pos.real, start_pos.imag))
                self._closed = True
                current_pos = start_pos
                command = None

            elif command == 'L':
                x = elements.pop()
                y = elements.pop()
                pos = float(x) + float(y) * 1j
                if not absolute:
                    pos += current_pos
                coord_points.append((pos.real, pos.imag))
                current_pos = pos

            elif command == 'H':
                x = elements.pop()
                pos = float(x) + current_pos.imag * 1j
                if not absolute:
                    pos += current_pos.real
                coord_points.append((pos.real, pos.imag))
                current_pos = pos

            elif command == 'V':
                y = elements.pop()
                pos = current_pos.real + float(y) * 1j
                if not absolute:
                    pos += current_pos.imag * 1j
                coord_points.append((pos.real, pos.imag))
                current_pos = pos

            elif command == 'C':
                control1 = float(elements.pop()) + float(elements.pop()) * 1j
                control2 = float(elements.pop()) + float(elements.pop()) * 1j
                end = float(elements.pop()) + float(elements.pop()) * 1j

                if not absolute:
                    control1 += current_pos
                    control2 += current_pos
                    end += current_pos

                coord_points.append((control1.real, control1.imag))
                coord_points.append((control2.real, control2.imag))
                coord_points.append((end.real, end.imag))
                current_pos = end

            elif command == 'S':
                # Smooth curve. First control point is the "reflection" of
                # the second control point in the previous path.

                if last_command not in 'CS':
                    # If there is no previous command or if the previous command
                    # was not an C, c, S or s, assume the first control point is
                    # coincident with the current point.
                    control1 = current_pos
                else:
                    # The first control point is assumed to be the reflection of
                    # the second control point on the previous command relative
                    # to the current point.
                    control1 = current_pos + current_pos - coord_points[-2]

                control2 = float(elements.pop()) + float(elements.pop()) * 1j
                end = float(elements.pop()) + float(elements.pop()) * 1j

                if not absolute:
                    control2 += current_pos
                    end += current_pos

                coord_points.append((control1.real, control1.imag))
                coord_points.append((control2.real, control2.imag))
                coord_points.append((end.real, end.imag))
                current_pos = end

            elif command == 'Q':
                control = float(elements.pop()) + float(elements.pop()) * 1j
                end = float(elements.pop()) + float(elements.pop()) * 1j

                if not absolute:
                    control += current_pos
                    end += current_pos

                coord_points.append((control.real, control.imag))
                coord_points.append((end.real, end.imag))
                current_pos = end

            elif command == 'T':
                # Smooth curve. Control point is the "reflection" of
                # the second control point in the previous path.

                if last_command not in 'QT':
                    # If there is no previous command or if the previous command
                    # was not an Q, q, T or t, assume the first control point is
                    # coincident with the current point.
                    control = current_pos
                else:
                    # The control point is assumed to be the reflection of
                    # the control point on the previous command relative
                    # to the current point.
                    control = current_pos + current_pos - coord_points[-2]

                end = float(elements.pop()) + float(elements.pop()) * 1j

                if not absolute:
                    end += current_pos

                coord_points.append((control.real, control.imag))
                coord_points.append((end.real, end.imag))
                current_pos = end

            elif command == 'A':

                radius = float(elements.pop()) + float(elements.pop()) * 1j
                rotation = float(elements.pop())
                arc = float(elements.pop())
                sweep = float(elements.pop())
                end = float(elements.pop()) + float(elements.pop()) * 1j

                if not absolute:
                    end += current_pos

                if radius.real == 0 or radius.imag == 0:
                    # Note: In browsers AFAIK, zero radius arcs are displayed
                    # as lines (see "examples/zero-radius-arcs.svg").
                    # Thus zero radius arcs are substituted for lines here.
                    self.msg('Replacing degenerate (zero radius) Arc with a Line: '
                         'Arc(start={}, radius={}, rotation={}, large_arc={}, '
                         'sweep={}, end={})'.format(
                        current_pos, radius, rotation, arc, sweep, end) +
                         ' --> Line(start={}, end={})'
                         ''.format(current_pos, end))
                    coord_points.append((end.real, end.imag))
                else:
                    coord_points.append((end.real, end.imag))
                current_pos = end
        return coord_points

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

        self.msg(f"Transform: {svg_image.get('transform')}")
        self.msg(f"Image offsets: (x: {x_offset}, y: {y_offset})")
        return (x_offset, y_offset)

    def get_max_dpi(self, svg_images_with_paths):
        max_dpi = 0 #NOTE: The "i" is a stand-in, this functions for any base svg measurement system
        for svg_image_with_path in svg_images_with_paths:
            max_dpi = max(max_dpi, svg_image_with_path['img_dpi'])

        return {"max_dpi": max_dpi}


    def get_overall_images_dims_offsets(self, svg_images_with_paths):
        x_min, x_max, y_min, y_max = 99999.0, -99999.0, 99999.0, -99999.0
        for svg_image_with_path in svg_images_with_paths:
            x_cur_min, x_cur_max = (float(svg_image_with_path['x']),
                                    float(svg_image_with_path['x'] + svg_image_with_path['width']))
            y_cur_min, y_cur_max = (float(svg_image_with_path['y']),
                                    float(svg_image_with_path['y'] + svg_image_with_path['height']))

            x_min = min(x_min, x_cur_min)
            x_max = max(x_max, x_cur_max)
            y_min = min(y_min, y_cur_min)
            y_max = max(y_max, y_cur_max)

        x, y = x_min, y_min
        width, height = x_max - x_min, y_max - y_min
        return {"x": x, "y": y, "width": width, "height": height}

    # def get_overall_cv_images_dims_offsets(self, svg_images_with_paths):
    #     x_min, x_max, y_min, y_max = 9999999.0, -9999999.0, 9999999.0, -9999999.0
    #     max_dpi = 0 #NOTE: The "i" is a stand-in, this functions for any base svg measurement system
    #     for svg_image_with_path in svg_images_with_paths:
    #         x_cur_min, x_cur_max = (float(svg_image_with_path['x_cv']),
    #                                 float(svg_image_with_path['x_cv'] + svg_image_with_path['width_cv']))
    #         y_cur_min, y_cur_max = (float(svg_image_with_path['y_cv']),
    #                                 float(svg_image_with_path['y_cv'] + svg_image_with_path['height_cv']))
    #
    #         x_min = min(x_min, x_cur_min)
    #         x_max = max(x_max, x_cur_max)
    #         y_min = min(y_min, y_cur_min)
    #         y_max = max(y_max, y_cur_max)
    #
    #         img_dpi = svg_image_with_path['width_cv']/svg_image_with_path['width']
    #         max_dpi = max(max_dpi, img_dpi)
    #
    #     x, y = x_min, y_min
    #     width, height = x_max - x_min, y_max - y_min
    #     #NOTE: x_cv and y_cv should both be 0!
    #     return {"x_cv": x, "y_cv": y, "width_cv": width, "height_cv": height, "max_dpi":max_dpi}

    def det_img_and_focus_specs(self, svg_images_with_paths, detail_bounds, approx_trace_path_string, crop_box):
        img_focus_specs = []
        for svg_image_with_path in svg_images_with_paths:
            img_focus_specs.append(str(svg_image_with_path['image_path']))
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
        # Check if ellipse or rect
        if crop_box.tag == inkex.addNS('rect', 'svg'):
            # Get rectangle properties
            img_focus_specs.append(str(float(crop_box.attrib.get('x', 0))))
            img_focus_specs.append(str(float(crop_box.attrib.get('y', 0))))
            img_focus_specs.append(str(float(crop_box.attrib.get('width', 0))))
            img_focus_specs.append(str(float(crop_box.attrib.get('height', 0))))
        else:
            raise inkex.AbortExtension("Only rectangles are supported as crop box.")

        #TODO: If only this changes level=3 once config setting up modify with ext input screen open
        img_focus_specs.append(str(approx_trace_path_string))

        selection_info_df = pd.DataFrame(img_focus_specs, columns=['selection_info'])

        # Add the 'line' column from the index
        # selection_info_df['line'] = selection_info_df.index

        # Reset the index if you want 'line' to be the first column
        # selection_info_df = selection_info_df[['line', 'selection_info']]  # reorders columns

        return selection_info_df

    def form_approx_control_points_normalized(self, approx_ctrl_points:list, overall_images_dims_offsets, crop_box):
        #Convert ctrl points to img yx format
        approx_ctrl_points_nd = np.array([(p[1], p[0]) for p in approx_ctrl_points])

        #Determine image offsets
        if crop_box is None:
            offsets_nd = np.array((overall_images_dims_offsets['y'], overall_images_dims_offsets['x']))
            norms = np.array((float(overall_images_dims_offsets['height']),
                              float(overall_images_dims_offsets['width'])))
        else:
            offsets_nd = np.array((float(crop_box.get('y')), float(crop_box.get('x'))))
            norms = np.array((float(crop_box.get('height')),
                              float(crop_box.get('width'))))

        #Determined shifted and normalized points
        formed_points_nd = (approx_ctrl_points_nd - offsets_nd)/norms
        return formed_points_nd


    def form_focus_region_specs_normalized(self, detail_bounds, overall_images_dims_offsets):
        #Determine image offsets
        (x_offset, y_offset) = (overall_images_dims_offsets['x'], overall_images_dims_offsets['y'])
        (x_norm, y_norm) = (overall_images_dims_offsets['width'], overall_images_dims_offsets['height'])
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

    def form_crop_box_advanced(self, crop_box, overall_images_dims_offsets):
        if crop_box is None:
            #Set points to overall images dims
            x_raw, y_raw, width_raw, height_raw = (overall_images_dims_offsets['x'], overall_images_dims_offsets['y'],
                                                   overall_images_dims_offsets['width'],
                                                   overall_images_dims_offsets['height'])
        else:
            x_raw, y_raw, width_raw, height_raw = (crop_box.get('x'), crop_box.get('y'),
                                                   float(crop_box.get('width')), float(crop_box.get('height')))

            x_raw, y_raw = 0.0 if x_raw is None else float(x_raw), 0.0 if y_raw is None else float(y_raw)

        #Set in svg crop-box dims
        adv_bound_points = {'x': x_raw, 'y': y_raw, 'width': width_raw, 'height': height_raw}
        adv_bound_points.update({'x_min': x_raw, 'y_min': y_raw,
                                 'x_max': x_raw + width_raw, 'y_max': y_raw + height_raw})

        # #Get normalized bound points
        # #NOTE: these are all relative to overall images offsets!!
        # (x_offset, y_offset) = (overall_images_dims_offsets['x'], overall_images_dims_offsets['y'])
        # (x_norm, y_norm) = (overall_images_dims_offsets['width'], overall_images_dims_offsets['height'])
        # x_min, y_min = (x_raw - x_offset) / x_norm, (y_raw - y_offset) / y_norm
        # x_max, y_max = x_min + width_raw / x_norm, y_min + height_raw / y_norm
        # x_min, y_min = max(x_min, 0.0), max(y_min, 0.0)
        # x_max, y_max = min(x_max, 1.0), min(y_max, 1.0)

        # #Get cv bound points
        # adv_bound_points['x_min_cv'] = int(round(x_min * overall_images_dims_offsets['width_cv'],0))
        # adv_bound_points['y_min_cv'] = int(round(y_min * overall_images_dims_offsets['height_cv'],0))
        # adv_bound_points['x_max_cv'] = int(round(x_max * overall_images_dims_offsets['width_cv'],0))
        # adv_bound_points['y_max_cv'] = int(round(y_max * overall_images_dims_offsets['height_cv'],0))
        #
        # adv_bound_points['width_cv'] = adv_bound_points['x_max_cv'] - adv_bound_points['x_min_cv']
        # adv_bound_points['height_cv'] = adv_bound_points['y_max_cv'] - adv_bound_points['y_min_cv']

        return adv_bound_points

    def crop_svg_images(self, svg_images_with_paths, advanced_crop_box):
        for svg_image_with_path in svg_images_with_paths:
            #Determine svg crop points if appl
            svg_image_with_path['x_crop_min'] = max(advanced_crop_box['x_min'] - svg_image_with_path['x'], 0)
            svg_image_with_path['x_crop_max'] = min(advanced_crop_box['x_max'] - svg_image_with_path['x'],
                                                       svg_image_with_path['width'])
            svg_image_with_path['y_crop_min'] = max(advanced_crop_box['y_min'] - svg_image_with_path['y'], 0)
            svg_image_with_path['y_crop_max'] = min(advanced_crop_box['y_max'] - svg_image_with_path['y'],
                                                       svg_image_with_path['height'])
            if (svg_image_with_path['y_crop_max'] - svg_image_with_path['y_crop_min'] <= 0
                or svg_image_with_path['x_crop_max'] - svg_image_with_path['x_crop_min'] <= 0):
                svg_image_with_path['include'] = False
            else:
                svg_image_with_path['include'] = True

                x_min_cv, x_max_cv = (int(round(svg_image_with_path['img_dpi']*svg_image_with_path['x_crop_min'],0)),
                                      int(round(svg_image_with_path['img_dpi']*svg_image_with_path['x_crop_max'],0)))
                y_min_cv, y_max_cv = (int(round(svg_image_with_path['img_dpi']*svg_image_with_path['y_crop_min'],0)),
                                      int(round(svg_image_with_path['img_dpi']*svg_image_with_path['y_crop_max'],0)))
                self.msg(f"width_cv {svg_image_with_path['width_cv']}, height_cv {svg_image_with_path['height_cv']}")
                self.msg(f"xmincv {x_min_cv}, xmaxcv {x_max_cv}"
                         f"ymincv {y_min_cv}, ymaxcv {y_max_cv}")
                svg_image_with_path['img_cv_cropped'] = svg_image_with_path['img_cv'][y_min_cv:y_max_cv,
                                                        x_min_cv:x_max_cv]
                self.msg("Crop dims: " + str(svg_image_with_path['img_cv_cropped'].shape))
                self.msg("DPI: " + str(svg_image_with_path['img_dpi']))

                #Amount to offset cropped image to edge of crop box (if at all)
                svg_image_with_path['x_crop_box_offset'] = (max(advanced_crop_box['x_min'], svg_image_with_path['x'])
                                                           - advanced_crop_box['x_min'])
                svg_image_with_path['y_crop_box_offset'] = (max(advanced_crop_box['y_min'], svg_image_with_path['y'])
                                                           - advanced_crop_box['y_min'])

        svg_images_with_paths = [s for s in svg_images_with_paths if s['include']]
        return svg_images_with_paths

    def get_straight_line_points(self, path_string):
        """Extracts points from a straight line path string without using re."""
        path_coords = self._parse_path(path_string)
        # for i, coord in enumerate(path_coords):
        #     self.msg(f"Coord {i}: {coord}")

        return path_coords

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
        svg_images = self.svg.selection.filter(inkex.Image).values()
        detail_bounds= self.svg.selection.filter(inkex.Ellipse).values()
        crop_boxes = self.svg.selection.filter(inkex.Rectangle).values()

        if len(crop_boxes) > 1:
            self.msg("Only 1 rectangle, used for cropping bound box")
            data_ret.close_connection()
            return

        crop_box = None
        for box in crop_boxes:
            crop_box = box
            break

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

        match len(svg_images):
            case 0:
                if len(self.svg.selected) > 0:
                    self.msg("No images found in selection! Check if you selected a group instead.")
                else:
                    self.msg("No images found in document!")
            case _:
                svg_images_with_paths = []
                for svg_image in svg_images:
                    image_path = self.checkImagePath(svg_image)
                    img_is_embedded = False
                    if image_path is None:  # check if image is embedded or linked
                        image_path = svg_image.get('{http://www.w3.org/1999/xlink}href')
                        img_is_embedded = True
                    offsets_xy = self.get_image_offsets(svg_image)
                    svg_images_with_paths.append({"svg_image": svg_image, "img_is_embedded": img_is_embedded,
                                                  "image_path": image_path,
                                                  "height": float(svg_image.get('height')),
                                                  "width": float(svg_image.get('width')),
                                                  "x": offsets_xy[0], "y": offsets_xy[1]})

                #Check to verify selection/doc/params haven't changed since last run
                approx_trace = None
                for path in approx_traces:
                    approx_trace = path
                    break
                approx_trace_path_string = approx_trace.get('d')
                img_and_focus_specs_df = self.det_img_and_focus_specs(svg_images_with_paths, detail_bounds,
                                                                      approx_trace_path_string,
                                                                      crop_box).dropna()
                update_level = data_ret.get_selection_match_level(self, img_and_focus_specs_df, self.options)

                self.msg("post-get selection match update level: " + str(update_level))

                if update_level > 0:
                    #If selection matches, check if params have changed
                    update_level = min(self.check_level_to_update(data_ret), update_level)

                self.msg("update level: " + str(update_level))
                #TODO: Get object creation working maybe, might not be worth it???
                if update_level == 3: update_level = 2

                #Determine overall offsets and dims:
                overall_images_dims_offsets = self.get_overall_images_dims_offsets(svg_images_with_paths)

                #Get approx control points for final path
                approx_ctrl_points = self.get_straight_line_points(approx_trace_path_string)
                formed_normalized_ctrl_points_nd = self.form_approx_control_points_normalized(approx_ctrl_points,
                                                                                             overall_images_dims_offsets,
                                                                                              crop_box)

                self.msg("Constrain: " + str(self.options.constrain_slic_within_mask))

                #Build up image data
                #TODO: Push this back into level 1, cache?
                start = time.time_ns()
                import cv2
                for svg_image_with_paths in svg_images_with_paths:
                    if svg_image_with_paths['img_is_embedded']:
                        import base64
                        # find comma position
                        i = 0
                        while i < 40:
                            if svg_image_with_paths['image_path'][i] == ',':
                                break
                            i = i + 1
                        img_data = base64.b64decode(
                            svg_image_with_paths['image_path'][i + 1:len(svg_image_with_paths['image_path'])])
                        img_array = np.frombuffer(img_data, dtype=np.uint8)
                        img_cv = {'img_cv': cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)}
                    else:
                        img_cv = {'img_cv': cv2.imread(svg_image_with_paths['image_path'], cv2.IMREAD_UNCHANGED)}

                    img_cv['height_cv'], img_cv['width_cv'] = img_cv['img_cv'].shape[0], img_cv['img_cv'].shape[1]
                    img_cv['img_dpi'] = img_cv['width_cv']/svg_image_with_paths['width']
                    # img_cv['y_cv']= int(round(((svg_image_with_paths['y'] - overall_images_dims_offsets['y'])
                    #                  *(float(img_cv['height_cv'])/svg_image_with_paths['height'])), 0))
                    # img_cv['x_cv'] = int(round(((svg_image_with_paths['x'] - overall_images_dims_offsets['x'])
                    #                   *(float(img_cv['width_cv'])/svg_image_with_paths['width'])), 0))
                    svg_image_with_paths.update(img_cv)

                end = time.time_ns()
                self.msg("SVG images processing time: " + str((end - start) / 1000000) + " ms")

                start = time.time_ns()

                #Find dpi of working matte
                overall_images_dims_offsets.update(self.get_max_dpi(svg_images_with_paths))

                #Form up normalized focus regions
                normalized_focus_region_specs =\
                    self.form_focus_region_specs_normalized(detail_bounds, overall_images_dims_offsets)

                #Build crop bounding box and crop images
                advanced_crop_box = self.form_crop_box_advanced(crop_box, overall_images_dims_offsets)
                svg_images_with_paths = self.crop_svg_images(svg_images_with_paths, advanced_crop_box)

                end = time.time_ns()
                self.msg("SVG shapes processing time: " + str((end - start) / 1000000) + " ms")

                #Retrieve or calculate data as needed
                objects = {}
                match update_level:
                    case 0 | 1:
                        import helpers.build_objects as buildscr
                        import helpers.caching.set_data_by_level as setdb

                        #Retrieve Level 1 objects from db
                        _, dataframes = data_ret.retrieve_and_wipe_data(0)

                        #Levels 1-4 objects from scratch
                        buildscr.build_level_1_scratch(self, svg_images_with_paths, overall_images_dims_offsets,
                                                       normalized_focus_region_specs, advanced_crop_box,
                                                       self.options, objects)
                        buildscr.build_level_2_scratch(self, self.options, objects)
                        start=time.time_ns()
                        buildscr.build_level_3_scratch(self, self.options, objects, formed_normalized_ctrl_points_nd,
                          overall_images_dims_offsets, advanced_crop_box)
                        end = time.time_ns()
                        self.msg("TIMER: Level 3 processing time: " + str((end - start) / 1000000) + " ms")
                        buildscr.build_level_4_scratch(self, self.options, objects, overall_images_dims_offsets)
                        if self.options.preview:
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
                        buildscr.build_level_2_scratch(self, self.options, objects)
                        buildscr.build_level_3_scratch(self, self.options, objects, formed_normalized_ctrl_points_nd,
                          overall_images_dims_offsets, advanced_crop_box)
                        buildscr.build_level_4_scratch(self, self.options, objects, overall_images_dims_offsets)

                        if self.options.preview:
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
                        buildscr.build_level_3_scratch(self, self.options, objects, formed_normalized_ctrl_points_nd,
                          overall_images_dims_offsets, advanced_crop_box)
                        buildscr.build_level_4_scratch(self, self.options, objects ,overall_images_dims_offsets)

                        if self.options.preview:
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
                        buildscr.build_level_4_scratch(self, self.options, objects, overall_images_dims_offsets)

                        if self.options.preview:
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
                    command_strs = []
                    for path_obj in [objects['raw_path'], objects['formed_path']]:

                        formed_path_nd = np.array(path_obj)
                        # formed_path_nd = np.array(objects['formed_path'])
                        formed_path_xy = formed_path_nd[:, [1, 0]]

                        #Determine scaling & shifting
                        size_cv = (int(round(overall_images_dims_offsets['max_dpi'] * advanced_crop_box['width'], 0)),
                         int(round(overall_images_dims_offsets['max_dpi'] * advanced_crop_box['height'], 0)))
                        x_scale = size_cv[0] / float(advanced_crop_box['width'])
                        y_scale = size_cv[1] / float(advanced_crop_box['height'])
                        scale_nd = np.array([x_scale, y_scale])
                        shift_nd = np.array([advanced_crop_box['x'], advanced_crop_box['y']])

                        #Offset main contour to line up with crop box on svg
                        formed_path_shifted = (formed_path_xy/scale_nd + shift_nd).tolist()

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
                        command_strs.append(commands_str)

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
                        for i,  command_str in enumerate(command_strs):
                            # Add a new path element to the SVG
                            path_element = inkex.PathElement()
                            path_element.set('d', command_str)  # Set the path data
                            if i == 0:
                                path_element.style = {'stroke': 'grey', 'fill': 'none', 'stroke-width': '4'}
                            else:
                                path_element.style = {'stroke': 'black', 'fill': 'none', 'stroke-width': '7'}
                            self.svg.get_current_layer().add(path_element)

                        #Clear the database
                        data_ret.wipe_data()

        data_ret.close_connection()

if __name__ == '__main__':
    try:
        import inkscape_ExtensionDevTools

        inkscape_ExtensionDevTools.inkscape_run_debug()
    except:
        pass
    continuous_outline().run()