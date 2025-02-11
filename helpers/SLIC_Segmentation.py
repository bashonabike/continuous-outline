# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
from typing import List, Tuple

# testimg = "C:\\Users\\liamc\\PycharmProjects\\continuous-outline\\Trial-AI-Base-Images\\image_fx_(18).jpg"
# # construct the argument parser and parse the arguments
# # load the image and convert it to a floating point data type
# image = img_as_float(io.imread(testimg))
# # loop over the number of segments
# for numSegments in (10, 20):
# 	# apply SLIC and extract (approximately) the supplied number
# 	# of segments
# 	segments = slic(image, n_segments = numSegments, sigma = 5)
# 	# show the output of SLIC
# 	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
# 	ax = fig.add_subplot(1, 1, 1)
# 	ax.imshow(mark_boundaries(image, segments))
# 	plt.axis("off")
# # show the plots
# plt.show()

def mask_test_boundaries(img_path, split_contours):
	#NOTE: only work PNG with transparent bg, or where background is all white
	img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	mask = None
	if img.shape[2] == 4:  # Check if it has an alpha channel
		alpha = img[:, :, 3]  # Get the alpha channel
		mask = np.where(alpha > 0, 1, 0).astype(np.uint8)  # Create binary mask
	else:
		# Image has no alpha channel, treat white as background
		mask = np.where(img == np.max(img), 0, 1).astype(np.uint8)  # Create binary mask

	return find_contours_near_boundaries(split_contours, mask, tolerance=2), mask


def slic_image_test_boundaries(im_float, split_contours, num_segments:int =2):
	segments = None
	#TODO: maybe do 2 segmentations, one manifold one just for the outline, use manifold one to determine some details
	for num_segs in range(num_segments, num_segments+20):
		segments_trial = slic(im_float, n_segments=num_segs, sigma=5, enforce_connectivity=True)
		if np.max(segments_trial) > 1:
			segments = segments_trial
			break
	if segments is None: raise Exception("Segmentation failed, image too disperate for outlineing")

	return find_contours_near_boundaries(split_contours, segments, tolerance=2), segments

#
# x_scale = image_size[0] / float(svg_image.get('width'))
# y_scale = image_size[1] / float(svg_image.get('height'))



def is_contour_near_segment_boundary(contour: np.ndarray, segments: np.ndarray, tolerance: int = 10,
									 min_nodes_prox_pct = 0.1, min_nodes_in_row = 5) -> bool:
	"""
	Checks if a contour is near a segment boundary in a segmented image.

	Args:
		contour: A NumPy array of shape (N, 1, 2) representing the contour points.
		segments: A NumPy array representing the segmented image (e.g., from slic).
		tolerance: The maximum distance (in pixels) for a contour point to be
				   considered near a boundary.

	Returns:
		True if the contour is near a boundary, False otherwise.
	"""
	nodes_prox, nodes_tot = 0, len(contour.reshape(-1, 2))
	node_run_length, nodes_row_hit = 0, False
	for point in contour.reshape(-1, 2):  # Iterate through the contour points
		x, y = point
		x = int(x)  # Ensure that x and y are ints to index the segment array
		y = int(y)
		if x < 0 or x >= segments.shape[1] or y < 0 or y >= segments.shape[0]:
			continue  # Check that the point is within the segment array bounds

		seg_min, seg_max = 9999, -1

		#search in grid centered around point
		x_min, x_max = x - tolerance, x + tolerance
		if x_min < 0: x_min = 0
		if x_max >= segments.shape[1]: x_max = segments.shape[1] - 1
		y_min, y_max = y - tolerance, y + tolerance
		if y_min < 0: y_min = 0
		if y_max >= segments.shape[0]: y_max = segments.shape[0] - 1
		#Find min and max segment values
		for nx in range(x_min, x_max + 1):
			for ny in range(y_min, y_max + 1):
				cur_seg = segments[ny, nx]
				if cur_seg < seg_min: seg_min = cur_seg
				if cur_seg > seg_max: seg_max = cur_seg

		#If boundary crossing found, return true
		if seg_min != seg_max:
			nodes_prox += 1
			node_run_length += 1
		else:
			node_run_length = 0
		if node_run_length >= min_nodes_in_row: nodes_row_hit = True

	return float(nodes_prox) / nodes_tot >= min_nodes_prox_pct and nodes_row_hit

def find_contours_near_boundaries(contours: List[np.ndarray], segments: np.ndarray, tolerance: int = 2) -> List[
	np.ndarray]:
	"""Finds contours near segment boundaries in a segmented image.

	  Args:
		  contours: A list of NumPy arrays, where each array represents a contour.
		  segments: A NumPy array representing the segmented image.
		  tolerance: The maximum distance (in pixels) for a contour point to be
					 considered near a boundary.

	  Returns:
		  A list of contours that are near segment boundaries.
	"""

	#TODO: extend this to more than 2 segment types??
	#Maybe convolve
	# #Determine flat averaging sum of each pixel to see if it is near a boundary
	# yrtyrt = segments.astype(np.uint8)
	# edges_slic = cv2.Sobel(segments.astype(np.uint8), cv2.CV_8U, 0, 0, ksize=3)
	# test_canny = cv2.Canny(yrtyrt, 350, 400)
	# kernel = np.ones((tolerance*2+1, tolerance*2+1), dtype=np.float32)
	#
	# # Perform the convolution (using 'same' padding to keep output size the same)
	# convolved_image = scipy.signal.convolve2d(segments, kernel, mode='same', boundary='fill')
	# segment_in_region = np.zeros((segments.shape[0], segments.shape[1]), dtype=np.uint8)
	# for nx in range(convolved_image.shape[1]):
	# 	for ny in range(convolved_image.shape[0]):
	# 		window_pixels = min(nx + 1, tolerance*2+1)*min(ny + 1, tolerance*2+1)
	# 		if convolved_image[ny, nx]%window_pixels != 0: segment_in_region[ny, nx] = 1


	near_boundary_contours = []
	for contour in contours:
		if is_contour_near_segment_boundary(contour, segments, tolerance):
			near_boundary_contours.append(contour)
	return near_boundary_contours