import math

#NOTE: should be in radians, even spacing between quadrants (pi/2, pi/4, pi/8,....,etc)
directions_incr = math.pi/16
proximity_search_radius = 40
segment_length = 10
parallel_search_radius = 5
max_deflect_rad = 2*math.pi/3
maze_sections_across = 20
cluster_start_point_size = 11
section_saturation_satisfied = 0.5
saturation_termination = 0.4 #NOTE: all paths are double-counted due to nature of contours
need_to_steer_off_edge = 2*math.pi/3
edge_magnetism_look_ahead_sections = maze_sections_across//2
edge_magnetism_cutoff = 0.5

slic_regions = 6

dir_smoothing_size, dir_smoothing_sigma = 21, 2.0


#CALCULATED DO NOT TOUCH!!
rev_max_deflect_rad = 2 * math.pi - max_deflect_rad
#########################