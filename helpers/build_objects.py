#region Helpers
from logging import exception



def shift_contours(contours: list, shift_y: int, shift_x: int) -> list:
    import numpy as np
    contours_shifted = []
    for contour in contours:
        contour_nd = np.array(contour)
        contour_nd[:, 0] += shift_y
        contour_nd[:, 1] += shift_x
        contours_shifted.append(contour_nd.tolist())
    return contours_shifted

def shift_and_crop(outer_edges, outer_contours_yx, bounds_outer,
                   inner_edges, inner_contours_yx, bounds_inner,
                   detail_req_masks):
    import time
    import numpy as np

    start = time.time_ns()
    crop = (tuple([min(o, c) for o, c in zip(bounds_outer[0], bounds_inner[0])]),
            tuple([max(o, c) for o, c in zip(bounds_outer[1], bounds_inner[1])]))

    shift_y, shift_x = (-1) * crop[0][0], (-1) * crop[0][1]
    outer_edges_cropped = outer_edges[crop[0][0]:crop[1][0] + 1, crop[0][1]:crop[1][1] + 1]
    inner_edges_cropped = inner_edges[crop[0][0]:crop[1][0] + 1, crop[0][1]:crop[1][1] + 1]
    outer_contours_yx_cropped = shift_contours(outer_contours_yx, shift_y, shift_x)
    inner_contours_yx_cropped = shift_contours(inner_contours_yx, shift_y, shift_x)

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

def ellipse_mask(cx, cy, rx, ry, shape):
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
    cy_scaled, ry_scaled = cy * m, ry * m
    cx_scaled, rx_scaled = cx * n, rx * n
    x, y = np.meshgrid(np.arange(n), np.arange(m))

    # Calculate the normalized distance from the ellipse center
    distance = ((x - cx_scaled) / rx_scaled)**2 + ((y - cy_scaled) / ry_scaled)**2

    # Pixels inside the ellipse have a distance <= 1
    inside = distance <= 1

    return inside

def rect_mask(x, y, width, height, shape):
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
    xx, yy = np.meshgrid(np.arange(n), np.arange(m))

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
def build_level_1_scratch(img_cv, focus_regions, options, objects: dict):
    import cv2
    import numpy as np
    from skimage.util import img_as_float
    import helpers.edge_detect.SLIC_Segmentation as slic
    import time

    im_unch = img_cv

    # im_bgrem = bgrem.remove_background(im_unch, "jit")

    if im_unch.shape[2] == 4:  # Check if it has an alpha channel
      im_float = img_as_float(cv2.cvtColor(im_unch, cv2.COLOR_BGRA2BGR))  # Convert from BGRA to BGR
      alpha_mask = im_unch[:, :, 3] == 0
      im_float[alpha_mask] = [0.0, 0.0, 0.0]
    elif im_unch.shape[2] == 2:
      im_float = img_as_float(cv2.cvtColor(im_unch, cv2.COLOR_GRAY2BGR)) #Greyscale PNG
      alpha_mask = im_unch[:, :, 1] == 0
      im_float[alpha_mask] = [0.0, 0.0, 0.0]
    else:
      im_float = img_as_float(im_unch)  # If it does not have an alpha channel, just use the image directly.

    # near_boudaries_contours, segments = slic.slic_image_test_boundaries(im_float, split_contours)
    # near_boudaries_contours, segments = slic.mask_test_boundaries(image_path, split_contours)

    outer_edges, outer_contours_yx, mask, bounds_outer = slic.mask_boundary_edges(options, im_unch)
    inner_edges, inner_contours_yx, segments, num_segs, bounds_inner = slic.slic_image_boundary_edges(options, im_float,
                                                                                        num_segments=options.slic_regions,
                                                                                        enforce_connectivity=False,
                                                                                        contour_offset = len(outer_contours_yx))

    detail_req_masks = []
    for region in focus_regions:
        if region['form'] == "ellipse":
            detail_req_masks.append(ellipse_mask(region['cx'], region['cy'], region['rx'], region['ry'], outer_edges.shape))
        elif region['form'] == "rectangle":
            detail_req_masks.append(rect_mask(region['x'], region['y'], region['width'], region['height'], outer_edges.shape))
        else:
            raise exception("Invalid focus shape passed in: " + region['form'])

    (outer_edges_cropped, inner_edges_cropped,
     outer_contours_yx_cropped, inner_contours_yx_cropped,
     detail_req_masks_cropped,
     shift_y, shift_x) = shift_and_crop(outer_edges, outer_contours_yx, bounds_outer,
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


def build_level_2_scratch(options, objects: dict):
    from helpers.mazify.MazeSections import MazeSections
    from helpers.mazify.MazeAgent import MazeAgent

    #Set up sections & agent
    maze_sections = MazeSections(options, objects['outer_edges'], options.maze_sections_across,
                                 options.maze_sections_across, objects['detail_req_masks'])

    maze_agent = MazeAgent(options, objects['outer_edges'], objects['outer_contours'],
                           objects['inner_edges'], objects['inner_contours'],
                           maze_sections)

    # Set objects into dict
    inst_out_objects = objects_to_dict(["maze_sections", "maze_agent"])
    objects.update(inst_out_objects)

def build_level_3_scratch(options, objects: dict):
    #Trace n center
    raw_path_coords_cropped = objects['maze_agent'].run_round_trace(options.trace_technique)
    if len(raw_path_coords_cropped) > 0:
        raw_path = shift_contours([raw_path_coords_cropped], (-1)*objects['shift_y'], (-1)*objects['shift_x'])[0]
    else:
        raw_path=raw_path_coords_cropped
    # Set objects into dict
    inst_out_objects = objects_to_dict(["raw_path"])
    objects.update(inst_out_objects)

def build_level_4_scratch(options, objects: dict):
    if len(objects['raw_path']) == 0:
        formed_path = objects['raw_path']
    else:
        import helpers.post_proc.smooth_path as smooth
        import helpers.post_proc.path_cleanup as clean
        import helpers.post_proc.post_effects as fx

        #Process raw path
        remove_blips = clean.remove_inout(objects['raw_path'], 50, 100)
        dithered = fx.lfo_dither(remove_blips, 20, 1000, 3.0)

        simplified = smooth.simplify_line(dithered, tolerance=options.simplify_tolerance)

        formed_path = simplified

    # Set objects into dict
    inst_out_objects = objects_to_dict(["formed_path"])
    objects.update(inst_out_objects)
#endregion