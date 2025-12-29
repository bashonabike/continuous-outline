import numpy as np
import cv2
import copy as cp
# import vtracer as vt
# import autotrace as aut
from sklearn.cluster import KMeans
from svgpathtools import Path, Line, wsvg, CubicBezier
from svgpathtools import svg2paths2
import math

import matplotlib.pyplot as plt


# from autotrace import Bitmap, VectorFormat

def split_contour_by_accumulated_deflection(contour, angle_threshold=270):
    """
    Splits a contour into sub-contours when the accumulated angle of deflection
    exceeds the threshold.

    Args:
        contour: A NumPy array representing the contour points (x,y coordinates).
        angle_threshold: The maximum accumulated deflection angle in degrees.

    Returns:
        A list of sub-contours, where each sub-contour is a NumPy array
        of contour points.
    """

    sub_contours = []
    current_sub_contour = []
    accumulated_angle = 0
    prev_pt = None
    test = len(contour)

    if len(contour) == 1:
        sub_contours.append(contour)
    else:
        angle_pos, angle_prev_pos, angle_same_dir = True, True, 0
        for pt_raw in contour:
            pt = pt_raw[0]
            angle_prev_pos = angle_pos
            if prev_pt is not None:
                # Calculate angle between current and previous segments
                if len(current_sub_contour) > 1:
                    prevprev_pt = current_sub_contour[-2][0]
                    v1, v2 = prev_pt - prevprev_pt, pt - prev_pt
                    cross_product = np.cross(v1, v2)
                    dot_product = np.dot(v1, v2)
                    angle_rad = np.arctan2(cross_product, dot_product)
                    angle = np.degrees(angle_rad)
                else:
                    angle = 0

                # Accumulate deflection
                if angle < 0 or angle > 180:
                    angle_pos = True
                else:
                    angle_pos = False

                if angle_pos != angle_prev_pos: angle_same_dir = 0
                angle_same_dir += (angle + 180)%360-180

                accumulated_angle = 0.8*accumulated_angle + 0.2*angle

                # Check for deflection threshold
                if (abs(accumulated_angle) >= angle_threshold or
                         abs(angle_same_dir) > 0.7*angle_threshold ):
                    sub_contours.append(np.array(current_sub_contour))
                    current_sub_contour = [[prev_pt]]
                    accumulated_angle = 0  # Reset accumulated angle
                    angle_same_dir = 0

            current_sub_contour.append([pt])
            prev_pt = pt

        if current_sub_contour:
            sub_contours.append(np.array(current_sub_contour))

    return sub_contours
def extract_y_channel_manual(img):
  """
  Extracts the Y' channel from an RGB image manually.
  Should be more efficient than doign full YUV conversion

  Args:
    img: The input RGB image as a NumPy array.

  Returns:
    A NumPy array representing the Y' channel.
  """

  # # Get image dimensions
  # height, width, _ = img.shape
  #
  # # Create an empty array to store the Y' channel
  # y_channel = np.zeros((height, width), dtype=np.uint8)
  #
  # # Define the Y' calculation coefficients (ITU-R BT.601)
  # r_coeff = 0.299
  # g_coeff = 0.587
  # b_coeff = 0.114
  #
  # # Calculate Y' for each pixel
  # for i in range(height):
  #   for j in range(width):
  #     r, g, b = img[i, j]
  #     y_channel[i, j] = int(r * r_coeff + g * g_coeff + b * b_coeff)

  # TODO: speed test cmpr with
  img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
  # Extract the Y' channel
  y_channel = img_yuv[:, :, 0]
  return y_channel

def split_path_by_deflection(path, deflection_threshold=270, distance_threshold=5, max_path_nodes = 500):
    """
    Splits an SVG path into multiple paths if the accumulated angle of deflection
    exceeds the threshold within a specified distance.

    Args:
        path: An SVGPathTools Path object.
        deflection_threshold: The maximum accumulated deflection angle in degrees.
        distance_threshold: The distance along the path to check for deflection.

    Returns:
        A list of SVGPathTools Path objects, representing the split paths.
    """

    split_paths = []
    current_path = Path()
    accumulated_deflection = 0
    distance_traveled = 0
    prev_tangent = None
    path_nodes = 0

    for segment in path:
        path_nodes += 1
        if isinstance(segment, Line):
            # Calculate tangent for Line
            tangent = segment.end - segment.start
        elif isinstance(segment, CubicBezier):
            # Approximate tangent at the end point of CubicBezier
            # (More accurate tangent calculation for Bezier curves is possible)
            tangent = segment.end - segment.control2

        if prev_tangent is not None:
            # Calculate angle between tangents
            dot_product = tangent.real * prev_tangent.real + tangent.imag * prev_tangent.imag
            if (abs(tangent) * abs(prev_tangent)) > 0:
                try:
                    angle = math.acos(dot_product / (abs(tangent) * abs(prev_tangent)))
                except:
                    angle=math.pi
            else:
                angle = 0
            deflection = angle * 180 / math.pi

            # Accumulate deflection and update distance
            accumulated_deflection += deflection
            distance_traveled += segment.length()

            # Check for splitting condition
            if (path_nodes >= max_path_nodes or
                    (accumulated_deflection >= deflection_threshold and distance_traveled >= distance_threshold)):
                split_paths.append(current_path)
                current_path = Path()
                accumulated_deflection = 0
                distance_traveled = 0
                path_nodes = 0

        current_path.append(segment)
        prev_tangent = tangent

    # Add the remaining path segment
    if current_path:
        split_paths.append(current_path)

    return split_paths

def split_svg_file(input_file_path, output_file_path,
                   deflection_threshold=270, distance_threshold=5):
    """
    Loads an SVG file, splits each path based on deflection and distance,
    and saves the result to a new SVG file.

    Args:
        input_file_path: Path to the input SVG file.
        output_file_path: Path to the output SVG file.
        deflection_threshold: The maximum accumulated deflection angle in degrees.
        distance_threshold: The distance along the path to check for deflection.
    """

    paths, attributes, svg_attributes = svg2paths2(input_file_path)
    split_paths_list = []

    for path in paths:
        split_paths = split_path_by_deflection(path, deflection_threshold, distance_threshold)
        if len(split_paths_list) == 0:
            split_paths_list = cp.deepcopy(split_paths)
        else:
            split_paths_list.extend(split_paths)

    # Write the split paths to the output file
    wsvg(split_paths_list, attributes=attributes, filename=output_file_path)

def remove_short_edges(image, min_length=10):
  """
  Removes edges from a grayscale image that are shorter than the specified minimum length.

  Args:
    image: The input grayscale image as a NumPy array.
    min_length: The minimum length of an edge to be kept (in pixels).

  Returns:
    A new image with short edges removed.
  """

  # Find contours in the image
  contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Create a mask image
  mask = np.zeros_like(image)

  # Draw the longer contours onto the mask
  for contour in contours:
    if cv2.arcLength(contour, closed=True) >= min_length:
      cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

  # Apply the mask to the original image
  result = cv2.bitwise_and(image, mask)

  return result

def detect_edges(image_path):
    """
    Detects edges in an image using Laplacian of Gaussian (LoG) algorithm.

    Args:
        image_path: The path to the input image file.

    Returns:
        A tuple containing two NumPy arrays. The first array represents the edges detected in the image, and the second array represents the contours of the edges.
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    #option greyscale vs y channel
    #Result seems basically the same
    img_y = extract_y_channel_manual(img)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_LAB)
    edges_post_laplace = np.zeros(img_g.shape, dtype=np.uint8)

    # for channel in [l_channel, a_channel, b_channel]:
    for channel in [img_g]:

        # Remove noise by blurring with a Gaussian filter
        #TODO: play with sigma and kernal size
        #TODO: guassian blur before or after flattening to single channel
        img_blurred = cv2.GaussianBlur(channel, (5,5), 3)

        # cannyd = cv2.Canny(img_y, 350, 400)

        #TODO: Play with params
        #src	Source image.
    # dst	Destination image of the same size and the same number of channels as src .
    # ddepth	Desired depth of the destination image, see combinations.
    # ksize	Aperture size used to compute the second-derivative filters. See getDerivKernels for details. The size must be positive and odd.
    # scale	Optional scale factor for the computed Laplacian values. By default, no scaling is applied. See getDerivKernels for details.
    # delta	Optional delta value that is added to the results prior to storing them in dst .
    # borderType	Pixel extrapolation method, see BorderTypes. BORDER_WRAP is not supported.
        laplaced = cv2.Laplacian(img_blurred, -1, ksize=5)

        _, thresholded = cv2.threshold(laplaced, 127, 255, cv2.THRESH_TOZERO)

        _, binary_orig = cv2.threshold(thresholded, np.max(img) // 2, np.max(img), cv2.THRESH_BINARY)
        zeros_idx = binary_orig != 0
        edges_post_laplace[zeros_idx] = binary_orig[zeros_idx]


    postedge = edges_post_laplace


    #
    # if np.mean(laplaced) > 170:
    #     laplaced_prime = cv2.bitwise_not(thresholded)
    # else:
    #     laplaced_prime = thresholded

    # pxrem = spr.StrayPixelRemover(1, 30)
    # pixels_removed = pxrem.process(postedge)
    # trimmed = remove_short_edges(pixels_removed, min_length=500)
    #
    # # Create a structuring element (kernel) for dilation
    # dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #
    # # Perform dilation
    # broaden_edges = cv2.dilate(trimmed, dilation_kernel, iterations=2)
    #
    # #Thin edges
    # thin_edges = cv2.erode(broaden_edges, dilation_kernel, iterations=2)
    #
    # trimmed_thin = remove_short_edges(thin_edges, min_length=1000)

    # cv2.imwrite('edges.jpg', trimmed_thin)

    # pxrem = spr.StrayPixelRemover(1, 30)
    # pixels_removed = pxrem.process(postedge)
    # thinned_invert = ft.fastThin(pixels_removed)
    # _, thinned_raw = cv2.threshold(thinned_invert, 127, 255, cv2.THRESH_BINARY_INV)
    # # thinned_bool, px_rem_bool = thinned_raw.astype(bool), pixels_removed.astype(bool)
    # # thinned_corr_bool = (thinned_bool & px_rem_bool)
    # # thinned = thinned_corr_bool.astype(np.uint8)
    # thinned = thinned_raw
    # final = trimmed

    contours, _ = cv2.findContours(postedge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    split_contours = []
    for contour in contours:
        split_contours.extend(split_contour_by_accumulated_deflection(contour, angle_threshold=270))
    svg_paths = []
    for contour in split_contours:
        path_data = "M " + " ".join(["{},{}".format(x[0][0], x[0][1]) for x in contour])
        path = Path(path_data)
        if path.length() > 0:
            svg_paths.append(Path(path_data))

    # Write SVG file
    wsvg(svg_paths, filename='test2.svg')



    # Detect contours
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on a blank image
    # traced_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # iterImg = binary
    # contoursall = cp.deepcopy(contours)
    iters = 0
    # while np.mean(iterImg) > 0.01 * np.max(iterImg) and iters < 1000:
    #     iters += 1
    #     mask = np.zeros_like(img)
    #     cv2.drawContours(mask, contoursall, -1, (255), thickness=2000)
    #     # cv2.imshow('Traced Image', mask)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #
    #     img_without_contours = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    #     iterImg = cp.deepcopy(img_without_contours)
    #     # Detect contours
    #     _, binary2 = cv2.threshold(img_without_contours, 1, np.max(img_without_contours), cv2.THRESH_BINARY)
    #     contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contoursall = contoursall + contours2

    return postedge,split_contours

def k_means_clustering(image_path):
    """
    Performs K-means clustering on an image.

    Args:
        image_path: The path to the image file to be clustered.

    Returns:
        None

    Notes:
        This function writes a clustered image to a file named 'clustered.png'.
    """
    pic = plt.imread(image_path)
    pic_n = pic.reshape(pic.shape[0] * pic.shape[1], pic.shape[2])
    pic_n.shape
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pic_n)
    pic2show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
    plt.imsave('clustered.png', cluster_pic)

    #TODO: maybe try to fill in the clusters, segment as object?