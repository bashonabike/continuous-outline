# import the necessary packages
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import find_contours
from scipy import signal
import numpy as np
import cv2
from typing import List
import math
import matplotlib.pyplot as plt
# from shapely.geometry import LineString
# from simplification.cutil import simplify_coords
import time

import helpers.mazify.temp_options as options


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

def find_transition_nodes(regioned_img: np.ndarray):
	num_regions = regioned_img.max()
	#Encode each region in base 10 and run with ones 3x3
	encoded_regions = np.pow(10, regioned_img - 1).astype(np.uint64)
	kernel = np.ones((3, 3), np.uint64)
	convolved = signal.convolve2d(encoded_regions, kernel, mode='same', boundary='fill')

	#Break down to decode areas with 3 or more regions
	transitions_map = np.zeros(regioned_img.shape, dtype=np.uint64)
	regions_rem = convolved
	for region_mux in range(num_regions):
		region_div = int(math.floor(math.pow(10, (num_regions - 1) - region_mux)))
		regions_rem_new = regions_rem % region_div
		transitions_map += np.where(regions_rem_new != regions_rem, 1, 0).astype(np.uint64)
		regions_rem = regions_rem_new

	#Identify regions where there are 3 or more regions
	transition_nodes = np.where(transitions_map >= 3, True, False)

	# Create a perimeter mask for 2 or more and edge pixel
	perimeter_mask = np.zeros_like(transitions_map, dtype=bool)
	perimeter_mask[0, :] = True  # Top row
	perimeter_mask[-1, :] = True  # Bottom row
	perimeter_mask[:, 0] = True  # Left column
	perimeter_mask[:, -1] = True  # Right column
	full_mask = (transitions_map >= 2) & perimeter_mask

	transition_nodes = transition_nodes | full_mask
	return transition_nodes

def mask_test_boundaries(img_path, split_contours):
	#NOTE: only work PNG with transparent bg, or where background is all white
	img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	mask = None
	if img.shape[2] == 4:  # Check if it has an alpha channel
		alpha = img[:, :, 3]  # Get the alpha channel
		mask = np.where(alpha > 0, 1, 0).astype(np.uint8)  # Create binary mask
	else:
		# Image has no alpha channel, treat white as background
		grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		_, inverted = cv2.threshold(grey, 0.9*np.max(grey), 1, cv2.THRESH_BINARY)
		mask = np.where(inverted == 0, 1, 0).astype(np.uint8)  # Create binary mask


	return find_contours_near_boundaries(split_contours, mask, tolerance=2), mask

def pixel_map_from_edge_contours(shape, contours, offset_idx):
	edges_final = np.zeros(shape, dtype=np.uint16)
	for contour_idx in range(len(contours)):
		cv2.drawContours(edges_final, contours, contour_idx,
						 (contour_idx + 1 + offset_idx, contour_idx + 1 + offset_idx,
						  contour_idx + 1 + offset_idx))
	return edges_final

def mask_boundary_edges(img_unchanged):
	start = time.time_ns()
	#NOTE: only work PNG with transparent bg, or where background is all white
	mask = None
	if img_unchanged.shape[2] == 4:  # Check if it has an alpha channel
		alpha = img_unchanged[:, :, 3]  # Get the alpha channel
		mask = np.where(alpha > 0, 255, 0).astype(np.uint8)  # Create binary mask
	elif img_unchanged.shape[2] == 2:  # Check if it has an alpha channel
		alpha = img_unchanged[:, :, 1]  # Get the alpha channel
		mask = np.where(alpha > 0, 255, 0).astype(np.uint8)  # Create binary mask
	else:
		# Image has no alpha channel, treat white as background
		grey = cv2.cvtColor(img_unchanged, cv2.COLOR_BGR2GRAY)
		_, inverted = cv2.threshold(grey, 0.9*np.max(grey), 1, cv2.THRESH_BINARY)
		mask = np.where(inverted == 0, 255, 0).astype(np.uint8)  # Create binary mask

	#Blur n rebinarize n find edges
	mask_blurred = cv2.GaussianBlur(mask, (9,9), 8)



	_, edges_binary = cv2.threshold(mask_blurred, 10, 1, cv2.THRESH_BINARY)
	erode_kernel = np.ones((5, 5), np.uint8)
	eroded_edges = cv2.erode(edges_binary, erode_kernel, iterations=1)*255
	fill_contours, _ = cv2.findContours(eroded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	filled_mask = eroded_edges.copy()
	# Fill contours (holes)
	# for cnt in fill_contours:
	cv2.drawContours(filled_mask, fill_contours, -1, (255,255,255), thickness=cv2.FILLED)  # -1 fills the contour
	blank = np.zeros(img_unchanged.shape, dtype=np.uint8)
	edges = mark_boundaries(blank, filled_mask, color=tuple([255 for d in range(img_unchanged.shape[2])]),
							mode='outer').astype(np.uint8)[:,:,0]



	# _, edges_binary = cv2.threshold(edges, 127, 1, cv2.THRESH_BINARY)
	# edges_bool = edges_binary.astype(bool)

	#Set final contours contours on blank coded for reference
	final_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	final_contours = [c for c in final_contours if len(c) > options.outer_contour_length_cutoff]
	final_contours.sort(key=len, reverse=True)
	edges_final = np.zeros_like(edges).astype(np.uint16)
	for contour_idx in range(len(final_contours)):
		cv2.drawContours(edges_final, final_contours, contour_idx, (contour_idx + 1, contour_idx + 1,
																	contour_idx + 1))

	# Pop edge image
	# blank = np.zeros(img.shape, dtype=np.uint8)
	# edges = mark_boundaries(blank, mask, color=(255, 255, 255), mode='outer').astype(np.uint8)[:,:,0]
	# mask_contours, hierarchy  = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# simpl_contours = []
	# if hierarchy.shape[0] == 1:
	# 	simpl_contours.append(simplify_path_rdp(mask_contours, tolerance=1))
	# else:
	# 	for contour in mask_contours:
	# 		simpl_contours.append(simplify_path_rdp(contour, tolerance=1))
	# simple_contours_plotted = cv2.drawContours(blank, simpl_contours, -1, (255,255,255), 1)
	#
	# _, edges_binary = cv2.threshold(simple_contours_plotted[:, :, 0], 127, 1, cv2.THRESH_BINARY)
	# edges_bool = edges_binary.astype(bool)

	#Flip contours to yx to conform to cv standard
	flipped_contours = []
	min_y, min_x, max_y, max_x = 99999, 99999, -1, -1
	for c in final_contours:
		if c[0].size == 1:
			cur_contour = [c[1], c[0]]
		else:
			cur_contour = [[p[0][1], p[0][0]] for p in c]

		ys, xs = zip(*cur_contour)
		min_y, min_x = min(min_y, min(ys)), min(min_x, min(xs))
		max_y, max_x = max(max_y, max(ys)), max(max_x, max(xs))

		flipped_contours.append(cur_contour)

	end = time.time_ns()
	print(str((end - start)/1e6) + " ms to do mask stuff")
	return edges_final, flipped_contours,  mask, ((min_y, min_x), (max_y, max_x))

def slic_image_boundary_edges(im_float, num_segments:int =2, enforce_connectivity:bool = True, contour_offset = 0):
	segments = None
	num_segs_actual = -1
	start = time.time_ns()
	for num_segs in range(num_segments, num_segments+20):
		segments_trial = slic(im_float, n_segments=num_segs, sigma=5, enforce_connectivity=enforce_connectivity)
		if np.max(segments_trial) > 1:
			segments = segments_trial
			num_segs_actual = num_segs
			break
	if segments is None: raise Exception("Segmentation failed, image too disparate for outlining")
	end = time.time_ns()
	print(str((end - start)/1e6) + " ms to segment image with " + str(num_segs_actual) + " segments")

	#Determine contours
	start = time.time_ns()
	contours = []
	for segment_val in range(1, np.max(segments)+1):
		finder = np.where(segments == segment_val, 255, 0).astype(np.uint8)
		partial_contours, _ = cv2.findContours(finder, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours.extend(partial_contours)

	contours = [c for c in contours if len(c) > options.inner_contour_length_cutoff]

	#Plot contours on blank coded for reference
	edges = np.zeros_like(segments).astype(np.uint16)
	for contour_idx in range(len(contours)):
		cv2.drawContours(edges, contours, contour_idx, (contour_idx + 1 + contour_offset,
														contour_idx + 1 + contour_offset,
														contour_idx + 1 + contour_offset))
	end = time.time_ns()
	print(str((end - start)/1e6) + " ms to find contours with " + str(len(contours)) + " contours")

	#Wipe outside edge, false contours
	perimeter_mask = np.zeros_like(edges, dtype=bool)
	perimeter_mask[0, :] = True  # Top row
	perimeter_mask[-1, :] = True  # Bottom row
	perimeter_mask[:, 0] = True  # Left column
	perimeter_mask[:, -1] = True  # Right column
	edges[perimeter_mask] = 0

	#Pop edge image
	# blank = np.zeros(im_float.shape, dtype=np.uint8)
	#
	# edges = mark_boundaries(blank, segments, color=(255, 255, 255), mode='outer')
	# _, edges_binary = cv2.threshold(edges[:,:,0], 127, 1, cv2.THRESH_BINARY)
	# edges_bool = edges_binary.astype(bool)

	#TODO: Constrict from 2 wide to 1 wide??
	#Flip contours to yx to conform to cv standard
	flipped_contours = []
	min_y, min_x, max_y, max_x = 99999, 99999, -1, -1
	for c in contours:
		if c[0].size == 1:
			cur_contour = [c[1], c[0]]
		else:
			cur_contour = [[p[0][1], p[0][0]] for p in c]

		ys, xs = zip(*cur_contour)
		min_y, min_x = min(min_y, min(ys)), min(min_x, min(xs))
		max_y, max_x = max(max_y, max(ys)), max(max_x, max(xs))

		#Remove portion of contours along perimeter
		processed_contours = remove_perimeter_ghosting(cur_contour, edges.shape[0] - 1, edges.shape[1] - 1)

		flipped_contours.extend(processed_contours)

	flipped_contours = [c for c in flipped_contours if len(c) > options.inner_contour_length_cutoff]
	flipped_contours.sort(key=lambda p: len(p), reverse=True)

	return edges, flipped_contours, segments, num_segs_actual, ((min_y, min_x), (max_y, max_x))

def remove_perimeter_ghosting(points_list, max_y, max_x):
	"""
	Eliminates points where x or y is 0 or 1023 from a NumPy array.

	Args:
		points_array: A 2D NumPy array of (y, x) points.

	Returns:
		A new NumPy array containing the filtered points.
	"""
	# points_array = np.array(points_list)
	# y_coords = points_array[:, 0]
	# x_coords = points_array[:, 1]
	#
	# # Create boolean masks
	# y_mask = np.logical_and(y_coords != 0, y_coords != max_y)
	# x_mask = np.logical_and(x_coords != 0, x_coords != max_x)
	#
	# # Combine masks
	# combined_mask = np.logical_and(y_mask, x_mask)
	#
	# # Apply the mask
	# filtered_points = points_array[combined_mask]

	points_array = np.array(points_list)

	#Check for long flat stretches, may be ghosting
	#Pad with first point so distance array shape conforms
	diffs = np.diff(np.vstack((points_array, points_array[0])), axis=0)
	# Calculate the Manhattan distances
	distances = np.sum(np.abs(diffs), axis=1)
	valid_distances_mask = distances <= options.max_inner_path_seg_manhatten_length
	#If just last one fails, assume unclosed negate
	if not valid_distances_mask[-1] and valid_distances_mask[-2]: valid_distances_mask[-1] = True

	#Find splits in contour
	false_indices = np.where(valid_distances_mask == False)[0]

	if not false_indices.size:
		return [points_list]  # No False values found

	diffs = np.diff(false_indices)
	split_indices = np.where(diffs != 1)[0] + 1
	false_segments = [(seg[0], seg[-1]) for seg in np.split(false_indices, split_indices)]

	segments = []
	#Build true segs, don't include if only 1 point
	if false_segments[0][0] > 1:
		segments.append(points_list[0:false_segments[0][0]])

	for f in range(len(false_segments) - 1):
		if false_segments[f + 1][0] - false_segments[f][1] > 1:
			segments.append(points_list[false_segments[f][1] + 1:false_segments[f + 1][0]])

	if len(points_list) - 1 - false_segments[-1][-1] > 1:
		segments.append(points_list[false_segments[-1][1] + 1:len(points_list) - 1])



	return segments


def slic_image_test_boundaries(im_float, split_contours, num_segments:int =2, enforce_connectivity:bool = True):
	segments = None
	for num_segs in range(num_segments, num_segments+20):
		segments_trial = slic(im_float, n_segments=num_segs, sigma=5, enforce_connectivity=enforce_connectivity)
		if np.max(segments_trial) > 1:
			segments = segments_trial
			break
	if segments is None: raise Exception("Segmentation failed, image too disparate for outlining")

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

def shift_contours(contours: list, shift_x: int, shift_y: int) -> list:
	contours_shifted = []
	for contour in contours:
		contour_nd = np.array(contour)
		contour_nd[:, 0] += shift_x
		contour_nd[:, 1] += shift_y
		contours_shifted.append(contour_nd.tolist())
	return contours_shifted

# def simplify_path_rdp(points, tolerance=1.0):
# 	"""Simplifies a path using the Ramer-Douglas-Peucker algorithm."""
# 	testt = points[:,0,:]
# 	tyuyt = ""
# 	line = LineString(points[:,0,:])
# 	simplified_coords = simplify_coords(line.coords, tolerance)
# 	return list(simplified_coords)