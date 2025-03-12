def remove_inout(path, manhatten_dist_thresh=0, max_blip_coords=50):
    #Find inouts
    removal_stretches = []
    removal_on = False
    removal_end_point = -1
    for c in range(len(path)):
        if not removal_on and c >= 2 and path[c - 2] == path[c]:
            removal_on = True
            removal_end_point = c - 1
        elif removal_on:
            if c - removal_end_point > max_blip_coords:
                removal_on = False
            #Check if inout is over
            elif removal_end_point - c < 0 or (abs(path[c][0] - path[removal_end_point - c][0]) +
                abs(path[c][1] - path[removal_end_point - c][1]) > manhatten_dist_thresh):
                removal_stretches.append((2*removal_end_point - (c - 1), c - 1))
                removal_on = False
    if len(removal_stretches) == 0:
        return path

    #Re-build path without inouts
    processed_path = path[0:removal_stretches[0][0]]
    for i in range(len(removal_stretches)):
        if removal_stretches[i][1] > len(path) - 1: break
        #NOTE: including 1 of the removed boundary so doesn't chop up path too much
        startpoint, endpoint = (removal_stretches[i][1],
                                removal_stretches[i + 1][0] if i < len(removal_stretches) - 1 else len(path))
        processed_path.extend(path[startpoint:endpoint])

    return processed_path