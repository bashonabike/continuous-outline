import math

#NOTE: should be in radians, even spacing between quadrants (pi/2, pi/4, pi/8,....,etc)
directions_incr = math.pi/16
proximity_search_radius = 40
segment_length = 10
parallel_search_radius = 5
max_deflect_rad = 2*math.pi/3
maze_sections_across = 30
cluster_start_point_size = 11
section_saturation_satisfied = 0.5
saturation_termination = 0.4 #NOTE: all paths are double-counted due to nature of contours
need_to_steer_off_edge = 2*math.pi/3
edge_magnetism_look_ahead_sections = maze_sections_across//2
edge_magnetism_cutoff = 0.5

slic_regions = 8

dir_smoothing_size, dir_smoothing_sigma = 21, 2.0

dumb_node_optional_weight = 4
dumb_node_min_opt_weight_reduced = 1
dumb_node_blank_weight = 10
dumb_node_required_weight = 1

section_tracker_max_walk = 6

outer_contour_length_cutoff = 100


#CALCULATED DO NOT TOUCH!!
rev_max_deflect_rad = 2 * math.pi - max_deflect_rad
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
