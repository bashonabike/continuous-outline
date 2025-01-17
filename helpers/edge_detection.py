import numpy as np
import cv2
import copy as cp
import vtracer as vt
import autotrace as aut
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from autotrace import Bitmap, VectorFormat
from PIL import Image

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
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    #option greyscale vs y channel
    #Result seems basically the same
    img_y = extract_y_channel_manual(img)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove noise by blurring with a Gaussian filter
    #TODO: play with sigma and kernal size
    #TODO: guassian blur before or after flattening to single channel
    img_blurred = cv2.GaussianBlur(img_g, (3, 3), 0)

    cannyd = cv2.Canny(img_y, 350, 400)

    #TODO: Play with params
    #src	Source image.
# dst	Destination image of the same size and the same number of channels as src .
# ddepth	Desired depth of the destination image, see combinations.
# ksize	Aperture size used to compute the second-derivative filters. See getDerivKernels for details. The size must be positive and odd.
# scale	Optional scale factor for the computed Laplacian values. By default, no scaling is applied. See getDerivKernels for details.
# delta	Optional delta value that is added to the results prior to storing them in dst .
# borderType	Pixel extrapolation method, see BorderTypes. BORDER_WRAP is not supported.
    laplaced = cv2.Laplacian(cannyd, -1, ksize=3)




    if np.mean(laplaced) > 170:
        laplaced_prime = cv2.bitwise_not(laplaced)
    else:
        laplaced_prime = laplaced

    trimmed = remove_short_edges(laplaced_prime, min_length=100)

    # Create a structuring element (kernel) for dilation
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # Perform dilation
    broaden_edges = cv2.dilate(trimmed, dilation_kernel, iterations=2)

    #Thin edges
    thin_edges = cv2.erode(broaden_edges, dilation_kernel, iterations=2)

    trimmed_thin = remove_short_edges(thin_edges, min_length=1000)

    cv2.imwrite('edges.jpg', trimmed_thin)

    # Thresholding to get binary image
    _, binary = cv2.threshold(laplaced_prime, np.max(img) // 2, np.max(img), cv2.THRESH_BINARY)

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

    return binary

def vectorize_edgified_image(edges):
    # #vtracer.convert_image_to_svg_py(inp,
    # out,
    # colormode = 'color',        # ["color"] or "binary"
    # hierarchical = 'stacked',   # ["stacked"] or "cutout"
    # mode = 'spline',            # ["spline"] "polygon", or "none"
    # filter_speckle = 4,         # default: 4
    # color_precision = 6,        # default: 6
    # layer_difference = 16,      # default: 16
    # corner_threshold = 60,      # default: 60
    # length_threshold = 4.0,     # in [3.5, 10] default: 4.0
    # max_iterations = 10,        # default: 10
    # splice_threshold = 45,      # default: 45
    # path_precision = 3          # default: 8
    #                             )
    # # cv2.imwrite('edges.jpg', edges)
    vt.convert_image_to_svg_py('edges.jpg', 'test.svg', colormode='binary')
    temp = 1
    # Load an image.
    # image = np.asarray(Image.open("edges.jpg").convert("RGB"))
    #
    # # Create a bitmap.
    # bitmap = Bitmap(image)
    #
    # # Trace the bitmap.
    # vector = bitmap.trace(centerline=True)
    #
    # # Save the vector as an SVG.
    # vector.save("autotrace.svg")
    #
    # # Get an SVG as a byte string.
    # svg = vector.encode(VectorFormat.SVG)

def k_means_clustering(image_path):
    pic = plt.imread(image_path)
    pic_n = pic.reshape(pic.shape[0] * pic.shape[1], pic.shape[2])
    pic_n.shape
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pic_n)
    pic2show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
    plt.imsave('clustered.png', cluster_pic)

    #TODO: maybe try to fill in the clusters, segment as object?