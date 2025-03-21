import math


cuda_slic = False
slic_regions = 6
slic_max_image_resolution = 500
slic_lanczos = False
slic_snap_edges = True
slic_lab = True
slic_greyscale = False
transparancy_cutoff = 0.5
maze_sections_across = 70
constrain_slic_within_mask = False
dumb_node_optional_weight = 1
dumb_node_optional_max_variable_weight = 8
dumb_node_blank_weight = 200
dumb_node_opt_jump_weight = 4
dumb_node_req_jump_weight = 1
dumb_node_required_weight = 1
max_inner_path_seg_manhatten_length = 50
outer_contour_length_cutoff = 50
inner_contour_length_cutoff = 50
inner_contour_variable_weights = True
trace_inner_too = True
scorched_earth = True
scorched_earth_weight_multiplier = 5
simplify_tolerance = 1.0
preview = True
ids = []
selected_nodes = []
scale = 1.0
mask_retain_inner_transparencies = True


#NOTE: should be in radians, even spacing between quadrants (pi/2, pi/4, pi/8,....,etc)
# directions_incr = math.pi/16
# proximity_search_radius = 40
# segment_length = 10
# parallel_search_radius = 5
# max_deflect_rad = 2*math.pi/3
# maze_sections_across = 70
# cluster_start_point_size = 11
# section_saturation_satisfied = 0.5
# saturation_termination = 0.4 #NOTE: all paths are double-counted due to nature of contours
# need_to_steer_off_edge = 2*math.pi/3
# edge_magnetism_look_ahead_sections = maze_sections_across//2
# edge_magnetism_cutoff = 0.5

# slic_regions = 12
#
# # dir_smoothing_size, dir_smoothing_sigma = 21, 2.0
#
# dumb_node_optional_weight = 1
# dumb_node_optional_max_variable_weight = 6 #Turn on with inner_contour_variable_weights
# dumb_node_min_opt_weight_reduced = 1
# dumb_node_blank_weight = 200
# dumb_node_opt_jump_weight = 1 #NOTE: should be 1/2 cost of jump, since always jumps 2x to get to next path in section
# dumb_node_req_jump_weight = 1 #NOTE: should be 1/2 cost of jump, since always jumps 2x to get to next path in section
# dumb_node_required_weight = 1
#
# max_inner_path_seg_manhatten_length = 50
#
# outer_contour_length_cutoff = 200
# inner_contour_length_cutoff = 20
# inner_contour_variable_weights = True #Shorter contours get higher weight, thus less preferable
#
# scorched_earth = True
# scorched_earth_weight_multiplier = 6
# #TODO: Make sure to reset scorched earth weights when generating new maze
#
# snake_trace_max_jump_from_outer = 2
# snake_details_polygon_faces = 7
#
# typewriter_lines = 5 #MUST BE AT LEAST 2!!!
# typewriter_traverse_threshold = 0.5 #Ratio of region height to typewriter line height
#
# zigzag_typewriter_lines = 5 #MUST BE AT LEAST 2!!!
# zigzag_typewriter_traverse_threshold = 0.3 #Ratio of region height to typewriter line height
#
# vertical_zigzag_lines = 5 #MUST BE AT LEAST 2!!!
# vertical_zigzag_traverse_threshold = 0.3 #Ratio of region height to typewriter line height
#
# back_forth_lines = 12 #MUST BE AT LEAST 2!!!
# back_forth_traverse_threshold = 0.3 #Ratio of region height to typewriter line height

#TODO: Do "include details proximal to outer edge" setting, maybe include edges that are at least a certain % within the allowable margin
#Set them as required

simplify_tolerance = 0.7


#CALCULATED DO NOT TOUCH!!
# rev_max_deflect_rad = 2 * math.pi - max_deflect_rad
#########################


#Maybe auto-discard mask edges below certain # of nodes
#Det sub-paths per section, set option % fill details
#Find best path fullfilling approx % fill details and satisfying % outer path coverage, minimizing connecting path lengths
#minimize intersections
#Pre-determine in each section shortest path between each sub-path
#Construct tree, options for each section+path combo
#Just look ahead into next section over following along each indiv path
#Node weight is cost (jump disp), weight whether outer or inner, disp deflection
#make sure to add counter-node jumping from other path too
#have fwd and backward defl
#Walk each path first determine walk nodes, then go thru sections det jump nodes
#No need to worry re intersections since paths never intersect
#Maybe start just closes point for jump, advance to maybe factor in sum of deflection

#Maybe just do shortest path, incorporate details circles as must haves
