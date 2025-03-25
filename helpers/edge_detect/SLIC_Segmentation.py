# import the necessary packages
# from skimage.measure import find_contours
import numpy as np
import cv2
# from typing import List
# import matplotlib.pyplot as plt
# from shapely.geometry import LineString
# from simplification.cutil import simplify_coords
import time

# import helpers.mazify.temp_options as options


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

def try_edge_snap(parent_inkex, image_orig: np.ndarray, contours, downscale_ratio, options):
	if downscale_ratio >= 1.0: return contours
	if downscale_ratio <= 0.0: raise Exception("Invalid downsample ratio")

	#Perform Laplacian edge detection & thresholding
	image_orig_int = (image_orig*255).astype(np.uint8)
	test = image_orig_int.max()
	if image_orig.ndim > 2 and image_orig.shape[2] > 1:
		img_gray = cv2.cvtColor(image_orig_int, cv2.COLOR_BGR2GRAY)
	else:
		img_gray = image_orig_int

	img_blurred = cv2.GaussianBlur(img_gray, (5,5), 3)
	laplaced = cv2.Laplacian(img_blurred, -1, ksize=5)
	edges = laplaced

	# edges = cv2.Canny(img_gray, 100, 200)

	_, thresholded = cv2.threshold(edges, 127, 255, cv2.THRESH_TOZERO)
	thresholded = thresholded.astype(np.bool_)

	#Determine distance threshold for snapping based on downscale ratio
	distance_threshold = 5.0/downscale_ratio

	#Iterate over contours, snapping where viable
	contours_snapped = []
	total_snapped = 0
	for contour in contours:
		contour_points = contour.reshape(-1, 2)  # Reshape to (n_points, 2)
		min_x, max_x  = np.min(contour_points[:, 0]), np.max(contour_points[:, 0])
		min_y, max_y = np.min(contour_points[:, 1]), np.max(contour_points[:, 1])
		thresholded_area = thresholded[max(int(min_y - distance_threshold), 0):int(max_y + distance_threshold),
						   max(int(min_x - distance_threshold), 0):int(max_x + distance_threshold)]  # Apply distance threshold:max_y, min_x:max_x]
		true_coords = np.array(np.where(thresholded_area)).T[:, [1, 0]]  # Transpose for easier iteration, switch to x,y
		if true_coords.shape[0] == 0:
			contours_snapped.append(contour)
			continue

		# Calculate distances (vectorized)
		distances = np.linalg.norm(true_coords - contour_points[:, np.newaxis], axis=2)

		# Find closest True pixels (vectorized)
		nearest_indices = np.argmin(distances, axis=1)
		nearest_distances = distances[np.arange(len(contour_points)), nearest_indices]
		nearest_true_pixels = true_coords[nearest_indices]

		# Check distance threshold (vectorized)
		within_threshold = nearest_distances <= distance_threshold

		total_snapped += np.sum(within_threshold)

		# Replace contour points with nearest True pixels
		contour_points[within_threshold] = nearest_true_pixels[within_threshold]

		contours_snapped.append(contour_points.reshape(contour.shape))  # Reshape back to original contour shape

	return contours_snapped

def try_downsample_mask(parent_inkex, mask: np.ndarray, options):
	#Check if image needs to be downsampled
	shape, downsample_ratio = mask.shape, 1.0
	if shape[0] > options.slic_max_image_resolution:
		downsample_ratio = options.slic_max_image_resolution / shape[0]
	if shape[1] > options.slic_max_image_resolution and shape[1] > shape[0]:
		downsample_ratio = options.slic_max_image_resolution / shape[1]
	if downsample_ratio <= 0.0: raise Exception("Invalid downsample ratio")
	if downsample_ratio >= 1.0: return mask, 1.0

	#Downsample mask
	#NOTE: dsize is width, height so shape[1], shape[0]
	dsize = (int(round(mask.shape[1] * downsample_ratio, 0)), int(round(mask.shape[0] * downsample_ratio, 0)))
	interpolation_technique = cv2.INTER_LANCZOS4 if parent_inkex.options.slic_lanczos else cv2.INTER_CUBIC
	downsampled_mask = cv2.resize(mask, dsize, interpolation=interpolation_technique).astype(np.bool_)

	return downsampled_mask, downsample_ratio

def try_downsample_and_smooth(parent_inkex, image: np.ndarray, options):
	#Check if image needs to be downsampled
	shape, downsample_ratio = image.shape, 1.0
	if shape[0] > options.slic_max_image_resolution:
		downsample_ratio = options.slic_max_image_resolution / shape[0]
	if shape[1] > options.slic_max_image_resolution and shape[1] > shape[0]:
		downsample_ratio = options.slic_max_image_resolution / shape[1]
	if downsample_ratio >= 1.0: return image, 1.0

	#Downsample and smooth
	#NOTE: dsize is width, height so shape[1], shape[0]
	start=time.time_ns()
	dsize = (int(round(shape[1] * downsample_ratio, 0)), int(round(shape[0] * downsample_ratio, 0)))
	interpolation_technique = cv2.INTER_LANCZOS4 if options.slic_lanczos else cv2.INTER_CUBIC
	downsampled_image = cv2.resize(image, dsize, interpolation=interpolation_technique)
	end=time.time_ns()
	print(str((end-start)/1e6) + " ms to downsample")

	start = time.time_ns()
	# 2. Smoothing (Gaussian blur)
	smoothed_image = cv2.GaussianBlur(downsampled_image, (5, 5), 0)  # Kernel size: 5x5, sigmaX=0

	# Or, smoothing (simple average blur)
	# smoothed_image = cv2.blur(downsampled_image, (5, 5))

	# Or, smoothing (median blur)
	# smoothed_image = cv2.medianBlur(downsampled_image, 5)
	end = time.time_ns()
	print(str((end - start) / 1e6) + " ms to smooth downsampled image")

	return smoothed_image, downsample_ratio

def try_upscale_contours(parent_inkex, contours, upsample_ratio):
	if upsample_ratio == 1.0: return contours

	#Upscale contours
	scaled_contours = []
	for contour in contours:
		contour_float = contour.astype(np.float32)
		contour_float *= upsample_ratio
		scaled_contours.append(contour_float.astype(np.int32))
	return scaled_contours

def find_transition_nodes(regioned_img: np.ndarray):
	from scipy import signal
	import math
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


def draw_and_flip_contours(inkex_parent, options, outer_contours, inner_contours, overall_images_dims_offsets,
						   advanced_crop_box):
	start = time.time_ns()
	import helpers.post_proc.smooth_path as smooth

	# outer_contours.sort(key=len, reverse=True)
	if len(inner_contours) > 0: inner_contours.sort(key=len, reverse=True)
	outer_edges = np.zeros((int(round(overall_images_dims_offsets['max_dpi']*advanced_crop_box['height'], 0)),
							int(round(overall_images_dims_offsets['max_dpi']*advanced_crop_box['width'], 0))),
						   dtype=np.uint16)
	inkex_parent.msg(f"Matte size: {outer_edges.shape}")
	inner_edges = outer_edges.copy()

	#Build perimeter mask
	perimeter_mask = np.zeros_like(outer_edges, dtype=bool)
	perimeter_mask[0, :] = True  # Top row
	perimeter_mask[-1, :] = True  # Bottom row
	perimeter_mask[:, 0] = True  # Left column
	perimeter_mask[:, -1] = True  # Right column

	#Draw on contours
	for contour_idx in range(len(outer_contours)):
		cv2.drawContours(outer_edges, outer_contours, contour_idx, (contour_idx + 1, contour_idx + 1,
																	contour_idx + 1))
	if len(inner_contours) > 0:
		for contour_idx in range(len(outer_contours), len(outer_contours) + len(inner_contours)):
			cv2.drawContours(inner_edges, inner_contours, contour_idx - len(outer_contours),
							 (contour_idx + 1, contour_idx + 1, contour_idx + 1))

	# Wipe outside edge, false contours
	outer_edges[perimeter_mask] = 0
	if len(inner_contours) > 0: inner_edges[perimeter_mask] = 0

	def flip_and_process_contours(contours):
		min_y, min_x, max_y, max_x = 99999, 99999, -1, -1
		flipped_contours = []

		for c in contours:
			if c[0].size == 1:
				cur_contour = smooth.simplify_line([c[1], c[0]], tolerance=options.simplify_tolerance,
												   preserve_topology=options.simplify_preserve_topology)
			else:
				cur_contour = smooth.simplify_line([[p[0][1], p[0][0]] for p in c],
												   tolerance=options.simplify_tolerance,
												   preserve_topology=options.simplify_preserve_topology)

			ys, xs = zip(*cur_contour)
			min_y, min_x = int(min(min_y, min(ys))), int(min(min_x, min(xs)))
			max_y, max_x = int(max(max_y, max(ys))), int(max(max_x, max(xs)))

			# Remove portion of contours along perimeter
			processed_contours = remove_perimeter_ghosting(options, cur_contour)

			flipped_contours.extend(processed_contours)

		bounds = ((min_y, min_x), (max_y, max_x))

		return flipped_contours, bounds


	# Flip contours to yx to conform to cv standard
	flipped_outer_contours, bounds_outer = flip_and_process_contours(outer_contours)
	if len(inner_contours) > 0:
		flipped_inner_contours, bounds_inner = flip_and_process_contours(inner_contours)
	else:
		flipped_inner_contours, bounds_inner = [], None

	end = time.time_ns()
	print(str((end - start) / 1e6) + " ms to draw n flip contours")
	return outer_edges, inner_edges, flipped_outer_contours, flipped_inner_contours, bounds_outer, bounds_inner

def mask_boundary_edges(parent_inkex, options, img_unchanged, overall_images_dims_offsets, svg_image_with_path):
	from skimage.segmentation import mark_boundaries
	import helpers.post_proc.smooth_path as smooth
	start = time.time_ns()
	#NOTE: only work PNG with transparent bg, or where background is all white
	mask = None
	if img_unchanged.shape[2] == 4:  # Check if it has an alpha channel
		alpha = img_unchanged[:, :, 3]  # Get the alpha channel
		max_alpha = np.max(alpha)
		transparancy_cutoff = options.transparancy_cutoff * max_alpha
		mask = np.where(alpha > transparancy_cutoff, 255, 0).astype(np.uint8)  # Create binary mask
	elif img_unchanged.shape[2] == 2:  # Check if it has an alpha channel
		alpha = img_unchanged[:, :, 1]  # Get the alpha channel
		max_alpha = np.max(alpha)
		transparancy_cutoff = options.transparancy_cutoff * max_alpha
		mask = np.where(alpha > transparancy_cutoff, 255, 0).astype(np.uint8)  # Create binary mask
	else:
		# Image has no alpha channel, treat white as background
		grey = cv2.cvtColor(img_unchanged, cv2.COLOR_BGR2GRAY)
		_, inverted = cv2.threshold(grey, (1.0-options.transparancy_cutoff)*np.max(grey), 1, cv2.THRESH_BINARY)
		mask = np.where(inverted == 0, 255, 0).astype(np.uint8)  # Create binary mask

	if options.mask_retain_inner_transparencies:
		if options.mask_retain_inner_erosion > 0:
			erosion_ksize = 2*((options.mask_retain_inner_erosion - 1)//2) + 1
			erode_kernel = np.ones((erosion_ksize, erosion_ksize), np.uint8)
			eroded_edges = cv2.erode(mask, erode_kernel, iterations=1)*255
			all_contours, _ = cv2.findContours(eroded_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		else:
			all_contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		parent_inkex.msg("num mask contours: " + str(len(all_contours)))
	else:
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
		all_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	start = time.time_ns()
	#Check length for removal
	final_contours_with_len = []
	for contour in all_contours:
		points = contour.reshape(-1, 2)
		diffs = np.diff(points, axis=0)
		distances = np.linalg.norm(diffs, axis=1)
		total_length = np.sum(distances)
		if total_length >= options.outer_contour_length_cutoff*svg_image_with_path['img_dpi']:
			final_contours_with_len.append({"contour":contour, "length":total_length})

	final_contours = [c['contour'] for c in sorted(final_contours_with_len,
												   key = lambda contour: contour['length'], reverse=True)]
	end = time.time_ns()
	parent_inkex.msg("TIMING: " + str((end - start) / 1e6) + " ms to remove too short outer contours")

	#Scale up to matte and offset final contours by prescribed offset amount
	max_dpi = overall_images_dims_offsets['max_dpi']
	scale = max_dpi/svg_image_with_path['img_dpi']
	x_cv_crop_box_offset, y_cv_crop_box_offset = (int(round(max_dpi*svg_image_with_path['x_crop_box_offset'],0)),
												  int(round(max_dpi*svg_image_with_path['y_crop_box_offset'],0)))
	for contour in final_contours:
		contour[:, :, 0] = np.round(scale*contour[:, :, 0]).astype(np.int32) + x_cv_crop_box_offset
		contour[:, :, 1] = np.round(scale*contour[:, :, 1]).astype(np.int32) + y_cv_crop_box_offset
		parent_inkex.msg(f"Max contour  x: {np.max(contour[:, :, 0])} y: {np.max(contour[:, :, 1])} for dpi {svg_image_with_path['img_dpi']}")

	complicated_background = len(final_contours) > 7
	parent_inkex.msg(f"{len(final_contours)} final contours")

	return final_contours,  mask, complicated_background

#C:\Users\liamc\PycharmProjects\continuous-outline\helpers\edge_detect\SLIC_Alg_Overview

def slic_image_boundary_edges(parent_inkex, options, im_float, mask, overall_images_dims_offsets, svg_image_with_path,
							  num_segments:int =2, enforce_connectivity:bool = True):
	if options.cuda_slic:
		from cuda_slic.slic import slic
	else:
		from skimage.segmentation import slic
	import helpers.post_proc.smooth_path as smooth
	segments = None
	num_segs_actual = -1
	start = time.time_ns()
	time_pre_updates = 0
	im_downscaled, downscale_ratio = try_downsample_and_smooth(parent_inkex, im_float, options)
	is_multichannel = im_downscaled.ndim > 2 and im_downscaled.shape[2] > 1
	for num_segs in range(num_segments, num_segments+20):
		if parent_inkex is not None: parent_inkex.msg("Trying " + str(num_segs) + " segments")
		if options.cuda_slic:
			segments_trial, time_pre_updates = slic(im_downscaled, max_iter=20, compactness=5, convert2lab=options.slic_lab,
													n_segments=num_segs, enforce_connectivity=enforce_connectivity,
													multichannel=is_multichannel)
		else:
			segments_trial, time_pre_updates = slic(im_downscaled,max_num_iter=20, compactness=5,
													convert2lab=options.slic_lab, n_segments=num_segs,
													sigma=5, enforce_connectivity=enforce_connectivity,
													channel_axis=-1 if is_multichannel else None)
		if np.max(segments_trial) > 1:
			segments = segments_trial
			num_segs_actual = num_segs
			break
	if segments is None: raise Exception("Segmentation failed, image too disparate for outlining")
	end = time.time_ns()
	if parent_inkex is not None: parent_inkex.msg(str((end - start)/1e6) + " ms to segment image with " + str(num_segs_actual) + " segments")

	if parent_inkex is not None: parent_inkex.msg(str(time_pre_updates) + " ms to do pre-cython prep of image slic")


	if options.constrain_slic_within_mask:
		#Blank outside of mask
		downsampled_mask, _ = try_downsample_mask(parent_inkex, mask, options)
		parent_inkex.msg(f"downsampled mask max: {downsampled_mask.max()} min: {downsampled_mask.min()}")
		segments[~downsampled_mask] = 0

	#Determine contours
	start = time.time_ns()
	contours = []
	#NOTE: Should have already thrown exception if downscale_ratio <= 0
	upscale_ratio = (1.0/downscale_ratio)*(overall_images_dims_offsets['max_dpi']/svg_image_with_path['img_dpi'])
	for segment_val in range(1, np.max(segments)+1):
		finder = np.where(segments == segment_val, 255, 0).astype(np.uint8)
		partial_contours, _ = cv2.findContours(finder, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		partial_contours_scaled = try_upscale_contours(parent_inkex, partial_contours, upscale_ratio)
		if options.slic_snap_edges:
			partial_contours_scaled = try_edge_snap(parent_inkex, im_float, partial_contours, downscale_ratio, options)
		contours.extend(partial_contours_scaled)

	#Check length for removal
	final_contours_with_len = []
	for contour in contours:
		points = contour.reshape(-1, 2)
		diffs = np.diff(points, axis=0)
		distances = np.linalg.norm(diffs, axis=1)
		total_length = np.sum(distances)
		#NOTE: Scale by max_dpi since contours already scaled up to max
		if total_length >= options.inner_contour_length_cutoff*overall_images_dims_offsets['max_dpi']:
			final_contours_with_len.append({"contour":contour, "length":total_length})

	final_contours = [c['contour'] for c in sorted(final_contours_with_len,
												   key = lambda contour: contour['length'], reverse=True)]

	#Offset final contours by prescribed offset amount
	max_dpi = overall_images_dims_offsets['max_dpi']
	x_cv_crop_box_offset, y_cv_crop_box_offset = (int(round(max_dpi*svg_image_with_path['x_crop_box_offset'],0)),
												  int(round(max_dpi*svg_image_with_path['y_crop_box_offset'],0)))
	for contour in final_contours:
		contour[:, :, 0] += x_cv_crop_box_offset
		contour[:, :, 1] += y_cv_crop_box_offset

	return final_contours, segments, num_segs_actual

def canny_hull_image_boundary_edges(img_unchanged, overall_images_dims_offsets, svg_image_with_path):
	"""
	Opens an image file using OpenCV, finds contours, and returns them.

	Args:
		image_path: The path to the image file.

	Returns:
		A list of contours found in the image, or None if an error occurs.
	"""

	# Convert to grayscale
	gray = cv2.cvtColor(img_unchanged, cv2.COLOR_BGR2GRAY)
	if img_unchanged.shape[2] == 4:  # Check if it has an alpha channel
		gray[img_unchanged[:,:,3] == 0] = 0

	gray = 5*(gray//5)

	# # Apply thresholding (you might need to adjust the threshold value)
	# _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	edges = cv2.Canny(gray, 100, 200)
	# Find contours
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	final_contours = []

	contours = [c for c in contours if len(c) > 100]
	for c in contours:
		contour = c[:,0,:]
		hull = cv2.convexHull(contour)
		area = cv2.contourArea(hull)
		perimeter = cv2.arcLength(hull, True)
		if area/perimeter > 10:
			final_contours.append(c)

	# Offset final contours by prescribed offset amount
	max_dpi = overall_images_dims_offsets['max_dpi']
	x_cv_crop_box_offset, y_cv_crop_box_offset = (
	int(round(max_dpi * svg_image_with_path['x_crop_box_offset'], 0)),
	int(round(max_dpi * svg_image_with_path['y_crop_box_offset'], 0)))
	for contour in final_contours:
		contour[:, :, 0] += x_cv_crop_box_offset
		contour[:, :, 1] += y_cv_crop_box_offset

	return final_contours

def remove_perimeter_ghosting(options, points_list):
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

def find_contours_near_boundaries(contours: list[np.ndarray], segments: np.ndarray, tolerance: int = 2) -> list[
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