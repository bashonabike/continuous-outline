#region Helpers

def shift_contours(contours: list, shift_y: int, shift_x: int) -> list:
    import numpy as np
    contours_shifted = []
    for contour in contours:
        contour_nd = np.array(contour)
        contour_nd[:, 0] += shift_y
        contour_nd[:, 1] += shift_x
        contours_shifted.append(contour_nd.tolist())
    return contours_shifted

def shift_and_crop(parent_inkex, outer_edges, outer_contours_yx, bounds_outer,
                   inner_edges, inner_contours_yx, bounds_inner,
                   detail_req_masks):
    import time
    import numpy as np

    start = time.time_ns()
    if bounds_inner is not None:
        crop = (tuple([min(o, c) for o, c in zip(bounds_outer[0], bounds_inner[0])]),
                tuple([max(o, c) for o, c in zip(bounds_outer[1], bounds_inner[1])]))
    else:
        crop = bounds_outer
    parent_inkex.msg(f"crop: {crop}")

    shift_y, shift_x = (-1) * crop[0][0], (-1) * crop[0][1]
    outer_edges_cropped = outer_edges[crop[0][0]:crop[1][0] + 1, crop[0][1]:crop[1][1] + 1]
    inner_edges_cropped = inner_edges[crop[0][0]:crop[1][0] + 1, crop[0][1]:crop[1][1] + 1]
    outer_contours_yx_cropped = shift_contours(outer_contours_yx, shift_y, shift_x)
    if len(inner_contours_yx) > 0:
        inner_contours_yx_cropped = shift_contours(inner_contours_yx, shift_y, shift_x)
    else:
        inner_contours_yx_cropped = []

    edges = outer_edges + inner_edges

    detail_req_masks_cropped = []
    for req_mask in detail_req_masks:
        detail_req_masks_cropped.append(req_mask[crop[0][0]:crop[1][0] + 1, crop[0][1]:crop[1][1] + 1])


    # transition_nodes = slic.find_transition_nodes(segments)
    #
    edges_show = edges.astype(np.uint8) * 255
    end = time.time_ns()
    print(str((end - start) / 1e6) + " ms to do crop and shift stuff")

    return (outer_edges_cropped, inner_edges_cropped, outer_contours_yx_cropped, inner_contours_yx_cropped,
            detail_req_masks_cropped, shift_y, shift_x)

def ellipse_mask(parent_inkex, cx, cy, rx, ry, shape):
    """
    Determines which pixels in an m x n ndarray are contained in an ellipse.

    Args:
        cx (float): x-coordinate of the ellipse center.
        cy (float): y-coordinate of the ellipse center.
        rx (float): Radius of the ellipse along the x-axis.
        ry (float): Radius of the ellipse along the y-axis.
        shape (tuple): Shape of ndarray.

    Returns:
        numpy.ndarray: A boolean m x n ndarray (mask) where True indicates a pixel is inside the ellipse.
    """

    import numpy as np
    # Create coordinate grids, scale up mask
    m, n = shape
    parent_inkex.msg("shape: " + str(shape))
    cy_scaled, ry_scaled = cy * m, ry * m
    cx_scaled, rx_scaled = cx * n, rx * n
    x, y = np.meshgrid(np.arange(m), np.arange(n))

    parent_inkex.msg("cy scaled: " + str(cy_scaled))
    parent_inkex.msg("cx scaled: " + str(cx_scaled))
    parent_inkex.msg("ry scaled: " + str(ry_scaled))
    parent_inkex.msg("rx scaled: " + str(rx_scaled))

    # Calculate the normalized distance from the ellipse center
    distance = ((x - cx_scaled) / rx_scaled)**2 + ((y - cy_scaled) / ry_scaled)**2

    # Pixels inside the ellipse have a distance <= 1
    inside = distance <= 1
    if np.count_nonzero(inside) > 0:
        y_indices, x_indices = np.where(inside)
        parent_inkex.msg("min y: " + str(np.min(y_indices)))
        parent_inkex.msg("max y: " + str(np.max(y_indices)))
        parent_inkex.msg("min x: " + str(np.min(x_indices)))
        parent_inkex.msg("max x: " + str(np.max(x_indices)))


    return inside

def rect_mask(parent_inkex, x, y, width, height, shape):
    """
    Determines which pixels in an m x n ndarray are contained in a rectangle.

    Args:
        x (float): x-coordinate of the top-left corner of the rectangle.
        y (float): y-coordinate of the top-left corner of the rectangle.
        width (float): Width of the rectangle.
        height (float): Height of the rectangle.
        shape (tuple): Shape of ndarray.

    Returns:
        numpy.ndarray: A boolean m x n ndarray (mask) where True indicates a pixel is inside the rectangle.
    """

    import numpy as np
    # Create coordinate grids, scale up vals
    m, n = shape
    y_scaled, height_scaled = y * m, height * m
    x_scaled, width_scaled = x * n, width * n
    yy,xx = np.meshgrid(np.arange(m), np.arange(n))

    # Check if pixels are within the rectangle bounds
    inside = (xx >= x_scaled) & (xx < x_scaled + width_scaled) & (yy >= y_scaled) & (yy < y_scaled + height_scaled)

    return inside

def objects_to_dict(obj_names):
    """
    Creates a dictionary where keys are variable names and values are objects.

    Args:
        *args: Variable objects.
        **kwargs: Named variable objects.

    Returns:
        dict: A dictionary of variable names and objects.
    """
    import sys

    result = {}

    # # Handle positional arguments
    # frame = inspect.currentframe().f_back
    # names = frame.f_code.co_varnames[frame.f_code.co_argcount:]
    # for i, obj in enumerate(args):
    #     if i < len(names):
    #         result[names[i]] = obj
    #     else:
    #         result[f"arg_{i}"] = obj # if more args than names, arg_0, arg_1, etc.
    #
    # # Handle keyword arguments
    # result.update(kwargs)
    for name in obj_names:
        result[name] = sys._getframe(1).f_locals[name]

    return result

#endregion


#region Builders
def build_level_1_scratch(parent_inkex, svg_images_with_paths, overall_images_dims_offsets, focus_regions,
                          advanced_crop_box, options, objects: dict):
    import cv2
    import numpy as np
    from skimage.util import img_as_float
    import helpers.edge_detect.SLIC_Segmentation as slic
    import time

    #TODO: Configure blend mode
    # if not options.slic_multi_image_conjoin_processing:
    outer_contours, inner_contours = [], []
    start = time.time_ns()
    for svg_image_with_path in svg_images_with_paths:
        im_unch = svg_image_with_path['img_cv_cropped']

        # im_bgrem = bgrem.remove_background(im_unch, "jit")
        if im_unch.shape[2] == 4:  # Check if it has an alpha channel
            alpha_mask = im_unch[:, :, 3] == 0
        elif im_unch.shape[2] == 2:
            alpha_mask = im_unch[:, :, 1] == 0
        else:
            alpha_mask = None
            im_float = img_as_float(im_unch)  # If it does not have an alpha channel, just use the image directly.

        if options.slic_greyscale:
            if im_unch.shape[2] == 4:  # Check if it has an alpha channel
                im_float = img_as_float(cv2.cvtColor(im_unch, cv2.COLOR_BGRA2GRAY))
                im_float[alpha_mask] = 0.0
            elif im_unch.shape[2] == 2:
                im_float = img_as_float(im_unch[:,:,0]) #Greyscale PNG
                im_float[alpha_mask] = 0.0
        else:
            if im_unch.shape[2] == 4:  # Check if it has an alpha channel
                im_float = img_as_float(cv2.cvtColor(im_unch, cv2.COLOR_BGRA2BGR))
                im_float[alpha_mask] = [0.0, 0.0, 0.0]
            elif im_unch.shape[2] == 2:
                im_float = img_as_float(cv2.cvtColor(im_unch, cv2.COLOR_GRAY2BGR)) #Greyscale PNG
                im_float[alpha_mask] = [0.0, 0.0, 0.0]

        end=time.time_ns()
        parent_inkex.msg(f"Converting to float took {(end - start) / 1e6} ms")
        # near_boudaries_contours, segments = slic.slic_image_test_boundaries(im_float, split_contours)
        # near_boudaries_contours, segments = slic.mask_test_boundaries(image_path, split_contours)

        outer_contours_cur, mask, complicated_background = slic.mask_boundary_edges(parent_inkex, options, im_unch,
                                                                                    overall_images_dims_offsets,
                                                                                    svg_image_with_path)
        outer_contours.extend(outer_contours_cur)
        start = time.time_ns()
        do_slic = False
        if not options.mask_only:
            do_slic = True
        if options.mask_only_when_complicated_background and complicated_background:
            do_slic = False
        if do_slic:
            if not options.canny_hull:
                inner_contours_cur, segments, num_segs = slic.slic_image_boundary_edges(parent_inkex, options, im_float, mask,
                                                                                        overall_images_dims_offsets,
                                                                                        svg_image_with_path,
                                                                                    num_segments=options.slic_regions,
                                                                                    enforce_connectivity=False)
            else:
                inner_contours_cur = slic.canny_hull_image_boundary_edges(im_unch, overall_images_dims_offsets,
                                                                          svg_image_with_path)
            inner_contours.extend(inner_contours_cur)

    #Flip contours and fill into edge arrays
    (outer_edges, inner_edges,
     outer_contours_yx, inner_contours_yx,
     bounds_outer, bounds_inner) = slic.draw_and_flip_contours(parent_inkex, options, outer_contours,
                                                               inner_contours, overall_images_dims_offsets,
                                                               advanced_crop_box)

    end = time.time_ns()
    parent_inkex.msg(f"SLIC took {(end - start) / 1e6} ms")

    ###TEMP####
    import inkex
    for outer_cont in outer_contours + inner_contours:
        commands = []
        for i, point in enumerate(outer_cont):
            if i == 0:
                commands.append(['M', point[0]])  # Move to the first point
            else:
                commands.append(['L', point[0]])  # Line to the next point
            # self.msg(str(point))
        # commands.append(['Z'])  # Close path
        command_strings = [
            f"{cmd_type} {x},{y}" for cmd_type, (x, y) in commands
        ]
        commands_str = " ".join(command_strings)

        # Add a new path element to the SVG
        path_element = inkex.PathElement()
        path_element.set('d', commands_str)  # Set the path data
        path_element.style = {'stroke': 'blue', 'fill': 'none'}
        parent_inkex.svg.get_current_layer().add(path_element)
    #######################################



    detail_req_masks = []
    for region in focus_regions:
        if region['form'] == "ellipse":
            detail_req_mask = ellipse_mask(parent_inkex, region['cx'], region['cy'], region['rx'], region['ry'],
                                           outer_edges.shape)
            if np.count_nonzero(detail_req_mask) > 0:
                detail_req_masks.append(detail_req_mask)
        elif region['form'] == "rectangle":
            detail_req_mask = rect_mask(parent_inkex, region['x'], region['y'], region['width'], region['height'],
                                        outer_edges.shape)
            if np.count_nonzero(detail_req_mask) > 0:
                detail_req_masks.append(detail_req_mask)
        else:
            raise Exception("Invalid focus shape passed in: " + region['form'])

    (outer_edges_cropped, inner_edges_cropped,
     outer_contours_yx_cropped, inner_contours_yx_cropped,
     detail_req_masks_cropped,
     shift_y, shift_x) = shift_and_crop(parent_inkex, outer_edges, outer_contours_yx, bounds_outer,
                                        inner_edges,inner_contours_yx, bounds_inner,
                                        detail_req_masks)

    #Set objects into dict
    inst_out_objects = objects_to_dict([
    "outer_edges_cropped",
    "inner_edges_cropped",
    "outer_contours_yx_cropped",
    "inner_contours_yx_cropped",
    "shift_y",
    "shift_x",
    "detail_req_masks_cropped"])
    final_out_objs = {}
    for key, value in inst_out_objects.items():
        if key.endswith("_yx_cropped"):
            new_key = key[:-len("_yx_cropped")]  # Remove the suffix
        elif key.endswith("_cropped"):
            new_key = key[:-len("_cropped")]  # Remove the suffix
        else:
            new_key = key
        final_out_objs[new_key] = value
    objects.update(final_out_objs)


def build_level_2_scratch(parent_inkex, options, objects: dict):
    from helpers.mazify.MazeSections import MazeSections
    from helpers.mazify.MazeAgent import MazeAgent

    #Set up sections & agent
    maze_sections = MazeSections(options, objects['outer_edges'], options.maze_sections_across,
                                 options.maze_sections_across, objects['detail_req_masks'])

    maze_agent = MazeAgent(parent_inkex, options, objects['outer_edges'], objects['outer_contours'],
                           objects['inner_edges'], objects['inner_contours'],
                           maze_sections)

    # Set objects into dict
    inst_out_objects = objects_to_dict(["maze_sections", "maze_agent"])
    objects.update(inst_out_objects)



    ###TEMP####
    import inkex
    for grid_line in maze_sections.grid_lines:
        commands = []
        for i, point in enumerate(grid_line):
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

        # Add a new path element to the SVG
        path_element = inkex.PathElement()
        path_element.set('d', commands_str)  # Set the path data
        path_element.style = {'stroke': 'green', 'fill': 'none'}
        parent_inkex.svg.get_current_layer().add(path_element)
    #######################################

def build_level_3_scratch(parent_inkex, options, objects: dict, approx_normalized_ctrl_points,
                          overall_images_dims_offsets, advanced_crop_box):
    #Get img heignt and width
    img_height, img_width = (overall_images_dims_offsets['max_dpi']*advanced_crop_box['height'],
                             overall_images_dims_offsets['max_dpi']*advanced_crop_box['width'])

    #Scale & shift control points
    import numpy as np
    shift_nd = np.array([objects['shift_y'], objects['shift_x']])
    scale_nd = np.array([img_height, img_width])
    approx_ctrl_points_nd = (approx_normalized_ctrl_points*scale_nd) + shift_nd

    # for ctrl in approx_ctrl_points_nd.tolist():
    #     parent_inkex.msg(ctrl)
    #Trace n center
    raw_path_coords_cropped, section_path, raw_points = (
        objects['maze_agent'].run_round_trace_approx_path(parent_inkex, approx_ctrl_points_nd,
                                                          overall_images_dims_offsets['max_dpi']*options.max_magnet_lock_dist))
    if len(raw_path_coords_cropped) > 0:
        raw_path = shift_contours([raw_path_coords_cropped], (-1)*objects['shift_y'], (-1)*objects['shift_x'])[0]
    else:
        raw_path=raw_path_coords_cropped

    # Set objects into dict
    inst_out_objects = objects_to_dict(["raw_path", "img_height", "img_width"])
    objects.update(inst_out_objects)

    # parent_inkex.msg(raw_path)



    # ###TEMP####
    import inkex
    rough_points = [(s[1], s[0]) for s in approx_ctrl_points_nd.tolist()]
    preview_layer = inkex.etree.Element(inkex.addNS('g', 'svg'),
                                        None, nsmap=inkex.NSS)
    preview_layer.set(inkex.addNS('groupmode', 'inkscape'), 'layer')
    preview_layer.set(inkex.addNS('label', 'inkscape'), 'Preview')
    preview_layer.set(inkex.addNS('lock', 'inkscape'), 'true')
    preview_layer.set(inkex.addNS('insensitive', 'inkscape'), 'true')

    preview_group = inkex.etree.SubElement(preview_layer, inkex.addNS('g', 'svg'))
    preview_group.set('id', 'preview_group')  # give the group an id so it can be found later.
    preview_group.set(inkex.addNS('lock', 'inkscape'), 'true')  # give the group an id so it can be found later.

    commands = []
    for i, point in enumerate(rough_points):
        x, y = point[0], point[1]
        circle_style = 'fill:#000000;stroke:none;stroke-width:0.264583'
        el = inkex.Circle.new(center=(x, y), radius=2)
        el.style = circle_style
        parent_inkex.svg.get_current_layer().add(el)



    import inkex
    commands = []
    for i, point in enumerate(rough_points):
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

    # Add a new path element to the SVG
    path_element = inkex.PathElement()
    path_element.set('d', commands_str)  # Set the path data
    path_element.style = {'stroke': 'red', 'fill': 'none'}
    parent_inkex.svg.get_current_layer().add(path_element)
    # #######################################

def build_level_4_scratch(parent_inkex, options, objects: dict, overall_images_dims_offsets):
    if len(objects['raw_path']) == 0:
        formed_path = objects['raw_path']
    else:
        import helpers.post_proc.smooth_path as smooth
        import helpers.post_proc.path_cleanup as clean
        import helpers.post_proc.post_effects as fx

        #Process raw path
        remove_repeated = clean.remove_repeated_coords(objects['raw_path'])
        if options.blip_max_thickness > 0 and options.blip_acuteness_threshold > 0:
            remove_blips = clean.remove_inout(parent_inkex, remove_repeated,
                                              options.blip_max_thickness*overall_images_dims_offsets['max_dpi'],
                                              options.blip_acuteness_threshold)
        else:
            remove_blips = remove_repeated
        dithered = fx.lfo_dither(remove_blips, 20, 1000, 3.0)
        if options.simplify_intelligent_straighting_cutoff > 0:
            smart_simp = smooth.intelligent_simplify_line(parent_inkex, dithered, options.simplify_intelligent_straighting_cutoff,
                                                          options.simplify_intelligent_straighting_cutoff/10)
        else:
            smart_simp = dithered

        simplified = smooth.simplify_line(smart_simp, tolerance=options.simplify_tolerance,
                                          preserve_topology=options.simplify_preserve_topology)

        formed_path = simplified

    # Set objects into dict
    inst_out_objects = objects_to_dict(["formed_path"])
    objects.update(inst_out_objects)
#endregion