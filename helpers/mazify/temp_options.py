import math

#NOTE: should be in radians, even spacing between quadrants (pi/2, pi/4, pi/8,....,etc)
directions_incr = math.pi/4
proximity_search_radius = 10
segment_length = 10
parallel_search_radius = 5
max_deflect_rad = math.pi/2
maze_sections_across = 4
cluster_start_point_size = 5
section_saturation_satisfied = 0.8
saturation_termination = 0.8


#CALCULATED DO NOT TOUCH!!
rev_max_deflect_rad = 2 * math.pi - max_deflect_rad
#########################