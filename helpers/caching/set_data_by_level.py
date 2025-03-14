import numpy as np
import pandas as pd


def set_level_1_data(dataframes:dict, input_data:dict):
    """
    Set the level 1 data for the contours and edges.
    :param outer_contours: dict of dataframes to set
    :param input_data: dict of objects to input
    :return: None
    """
    #Retrieve input data
    outer_contours = input_data["outer_contours"]
    outer_edges = input_data["outer_edges"]
    inner_contours = input_data["inner_contours"]
    inner_edges = input_data["inner_edges"]
    focus_masks = input_data["focus_masks"]


    #Set contours headers
    contour_idx_offset = 0
    for outer, contours in zip([True, False], [outer_contours, inner_contours]):
        num_rows = len(contours)
        contour_values = pd.Series(range(1 + contour_idx_offset, num_rows + contour_idx_offset + 1))
        is_outer_values = pd.Series([1] if outer else [0] * num_rows)
        is_inner_values = pd.Series([0] if outer else [1]  * num_rows)
        partial_df = pd.DataFrame({'contour': contour_values, 'is_outer': is_outer_values, 'is_inner': is_inner_values,
                                   'img_height': outer_edges.shape[0], 'img_width': outer_edges.shape[1]})
        partial_df = partial_df[dataframes["Contours"].columns]
        dataframes["Contours"] = pd.concat([dataframes["Contours"], partial_df], ignore_index=True)
        contour_idx_offset = num_rows


    #Set contours details
    for i, contour in enumerate(outer_contours + inner_contours):
        # contour_nums = pd.Series([i + 1] * len(contour))
        points = pd.Series(range(1, len(contour) + 1))
        contour_nd = np.array(contour)
        y_values = pd.Series(contour_nd[:, 0])
        x_values = pd.Series(contour_nd[:, 1])
        partial_df = pd.DataFrame({'point_num': points, 'x': x_values, 'y': y_values, 'contour': i + 1})
        partial_df = partial_df[dataframes["Contour"].columns]
        dataframes["Contour"] = pd.concat([dataframes["Contour"], partial_df], ignore_index=True)


    # #Set edges pixel maps
    # if outer_edges.shape != inner_edges.shape:
    #     raise ValueError("Outer and inner edge arrays must have the same shape.")
    #
    # y_coords, x_coords = np.indices(outer_edges.shape)  # Generate y and x coordinates using indices
    #
    # # Flatten the arrays
    # y_flat = y_coords.flatten()
    # x_flat = x_coords.flatten()
    # outer_flat = outer_edges.flatten()
    # inner_flat = inner_edges.flatten()
    #
    # # Create is_inner and is_outer columns
    # is_inner = np.where(inner_flat != 0, 1, 0)
    # is_outer = np.where(outer_flat != 0, 1, 0)
    #
    # # Create the DataFrame
    # partial_df = pd.DataFrame({
    #     'y': y_flat,
    #     'x': x_flat,
    #     'is_inner': is_inner,
    #     'is_outer': is_outer,
    #     'inner_edge_num': inner_flat,
    #     'outer_edge_num': outer_flat
    # })[dataframes["EdgesPixelMap"].columns]
    #
    # dataframes["EdgesPixelMap"] = partial_df


    #Set focus masks header
    focus_masks_sr = pd.Series(range(0, len(focus_masks)))
    partial_df = pd.DataFrame({'focus_mask': focus_masks_sr})
    dataframes["FocusMasks"] = partial_df


    #Set focus masks details
    for i, focus_mask in enumerate(focus_masks):
        y_coords, x_coords = np.where(focus_mask)

        # Create the DataFrame
        partial_df = pd.DataFrame({
            'y': y_coords,
            'x': x_coords,
            'focus_mask': i
        })[dataframes["FocusMask"].columns]

        dataframes["FocusMask"] = pd.concat([dataframes["FocusMask"], partial_df], ignore_index=True)

def set_level_2_data(dataframes:dict, input_data:dict):
    """
    Set the level 2 data for the contours and edges.
    :param outer_contours: dict of dataframes to set
    :param input_data: dict of objects to input
    :return: None
    """

    #Retrieve input data
    sections = input_data["sections"]
    agent = input_data["agent"]
    path_graph = sections.path_graph
    focus_sections = sections.focus_region_sections
    edge_paths = agent.all_contours_objects


    #Set sections header
    dataframes["Sections"] = pd.DataFrame({
        'height': [sections.m],
        'width': [sections.n],
        'y_grade': [sections.y_grade],
        'x_grade': [sections.x_grade],
        'img_height': [sections.img_height],
        'img_width': [sections.img_width]
    })[dataframes["Sections"].columns]


    #Set sections details
    y_coords, x_coords = np.indices(sections.sections.shape)  # Generate y and x coordinates using indices

    # Flatten the arrays
    y_flat = y_coords.flatten()
    x_flat = x_coords.flatten()
    sections_flat = sections.sections.flatten()

    dumb_req,dumb_opt,y_start,y_end,x_start,x_end,num_edge_pixels,is_focus_region, focus_region_nums = (
        zip(*[(s.dumb_req, s.dumb_opt, s.ymin, s.ymax, s.xmin, s.xmax, s.edge_pixels, s.focus_region,
               ','.join(str(n) for n in s.focus_region_nums))
              for s in sections_flat]))

    #Create dataframe
    dataframes["Section"] = pd.DataFrame({
        'y_sec': y_flat,
        'x_sec': x_flat,
        'dumb_req': dumb_req,
        'dumb_opt': dumb_opt,
        'y_start': y_start,
        'y_end': y_end,
        'x_start': x_start,
        'x_end': x_end,
        'num_edge_pixels': num_edge_pixels,
        'is_focus_region': is_focus_region,
        'focus_region_nums': focus_region_nums
    })[dataframes["Section"].columns]


    #Set focus sections
    for i, focus_mask in enumerate(focus_sections):
        y_sec, x_sec = zip(*[(s.y_sec, s.x_sec) for s in focus_mask])

        # Create the DataFrame
        partial_df = pd.DataFrame({
            'y_sec': y_sec,
            'x_sec': x_sec,
            'focus_mask': i
        })[dataframes["FocusSection"].columns]

        dataframes["FocusSection"] = pd.concat([dataframes["FocusSection"], partial_df], ignore_index=True)


    #Set edge paths
    path, is_outer, is_closed, custom_weight = zip(*[(p.num, p.outer, p.closed, p.custom_weight) for p in edge_paths])
    dataframes["EdgePath"] = pd.DataFrame({
        'path': path,
        'is_outer': is_outer,
        'is_closed': is_closed,
        'custom_weight': custom_weight
    })[dataframes["EdgePath"].columns]


    #Set edge nodes
    for edge_path in edge_paths:
        path = edge_path.path
        y, x, tracker_num, is_outer, y_sec, x_sec = zip(
            *[(n.y, n.x, n.section_tracker_num, n.outer, n.section.y_sec, n.section.x_sec) for n in path])
        node = range(0, len(path))
        partial_df = pd.DataFrame({
            'y_sec': y_sec,
            'x_sec': x_sec,
            'y': y,
            'x': x,
            'tracker_num': tracker_num,
            'is_outer': is_outer,
            'node': node,
            'path': edge_path.num
        })[dataframes["PathNode"].columns]
        dataframes["PathNode"] = pd.concat([dataframes["PathNode"], partial_df], ignore_index=True)

    #Set section trackers
    max_tracker_size, path_tracker_seq_offset = 0, 0
    for edge_path in edge_paths:
        trackers = edge_path.section_tracker
        if len(trackers) > max_tracker_size: max_tracker_size = len(trackers)
        in_node,out_node,path,rev_in_node,rev_out_node,tracker_num,prev_tracker,next_tracker,y_sec,x_sec = zip(*[(
            t.in_node.num,t.out_node.num,t.path_num,-1 if t.rev_in_node is None else t.rev_in_node.num,
            -1 if t.rev_out_node is None else t.rev_out_node.num,t.tracker_num,
            -1 if t.prev_tracker is None else t.prev_tracker.tracker_num,
            -1 if t.next_tracker is None else t.next_tracker.tracker_num,
            t.section.y_sec,t.section.x_sec) for t in trackers])
        path_tracker_seq = range(path_tracker_seq_offset + 0, path_tracker_seq_offset + len(trackers))
        partial_df = pd.DataFrame({
            'in_node': in_node,
            'out_node': out_node,
            'path': path,
            'rev_in_node': rev_in_node,
            'rev_out_node': rev_out_node,
            'tracker_num': tracker_num,
            'prev_tracker': prev_tracker,
            'next_tracker': next_tracker,
            'y_sec': y_sec,
            'x_sec': x_sec,
            'path_tracker_seq': path_tracker_seq
        })[dataframes["SectionTracker"].columns]
        dataframes["SectionTracker"] = pd.concat([dataframes["SectionTracker"], partial_df], ignore_index=True)
        path_tracker_seq_offset += len(trackers)


    #Set graph nodes
    nodes_with_cat = [(node, data.get('category')) for node, data in path_graph.nodes(data=True)]
    blank_nodes, blank_node_cats = zip(*[n for n in nodes_with_cat if len(n[0]) == 2])
    edged_nodes, edged_node_cats = zip(*[n for n in nodes_with_cat if len(n[0]) == 4])
    nodes = np.vstack((np.hstack((np.array(blank_nodes), np.zeros((len(blank_nodes), 2)))), np.array(edged_nodes)))
    cats = [c.value for c in blank_node_cats] + [c.value for c in edged_node_cats]

    #Figure out hashing
    height, width = sections.img_height, sections.img_width
    num_paths, num_trackers = len(edge_paths) + 1, max_tracker_size #NOTE: +1 since 1-indexed
    hash = (width * num_paths * num_trackers * nodes[:, 0]+ num_paths * num_trackers * nodes[:, 1] +
            num_trackers * nodes[:, 2] + nodes[:, 3])

    #Set into dataframe
    dataframes["GraphNode"] = pd.DataFrame({
        'graph_node': hash,
        'graph_node_y': nodes[:, 0],
        'graph_node_x': nodes[:, 1],
        'graph_node_path': nodes[:, 2],
        'graph_node_tracker': nodes[:, 3],
        'category': cats
    })[dataframes["GraphNode"].columns]

    #Set edges
    indices, in_nodes, out_nodes, weights = zip(*[(index,
                                                   (in_node[0], in_node[1], 0 if len(in_node) == 2 else in_node[2],
                                                    0 if len(in_node) == 2 else in_node[3]),
                                                   (out_node[0], out_node[1], 0 if len(out_node) == 2 else out_node[2],
                                                    0 if len(out_node) == 2 else out_node[3]), data.get('weight'))
                         for index, (in_node, out_node, data) in enumerate(path_graph.edges(data=True))])
    in_nodes_nd, out_nodes_nd = np.array(in_nodes), np.array(out_nodes)

    #Figure out hashing
    hash_in = (width * num_paths * num_trackers * in_nodes_nd[:, 0]+ num_paths * num_trackers * in_nodes_nd[:, 1] +
            num_trackers * in_nodes_nd[:, 2] + in_nodes_nd[:, 3])
    hash_out = (width * num_paths * num_trackers * out_nodes_nd[:, 0]+ num_paths * num_trackers * out_nodes_nd[:, 1] +
            num_trackers * out_nodes_nd[:, 2] + out_nodes_nd[:, 3])

    #Set into dataframe
    dataframes["GraphEdge"] = pd.DataFrame({
        'from_node': hash_in,
        'to_node': hash_out,
        'weight': weights
    })[dataframes["GraphEdge"].columns]


    #Set agent
    dataframes["Agent"] = pd.DataFrame({
        'start_node': [-1 if agent.start_node is None else agent.start_node.num],
        'end_node': [-1 if agent.end_node is None else agent.end_node.num],
        'start_path': [-1 if agent.start_node is None else agent.start_node.path_num],
        'end_path': [-1 if agent.end_node is None else agent.end_node.path_num],
        'max_tracker_size': [agent.max_tracker_size]
    })[dataframes["Agent"].columns]


def set_level_3_data(dataframes:dict, input_data:dict):
    """
    Set the level 2 data for the contours and edges.
    :param outer_contours: dict of dataframes to set
    :param input_data: dict of objects to input
    :return: None
    """

    #Retrieve raw path data
    raw_path = input_data["raw_path"]
    indices, y, x = zip(*[(index, n[0], n[1]) for index, n in enumerate(raw_path)])
    dataframes["RawPath"] = pd.DataFrame({
        'path_num': indices,
        'y': y,
        'x': x
    })[dataframes["RawPath"].columns]

def set_level_4_data(dataframes:dict, input_data:dict):
    """
    Set the level 2 data for the contours and edges.
    :param outer_contours: dict of dataframes to set
    :param input_data: dict of objects to input
    :return: None
    """

    #Retrieve raw path data
    formed_path = input_data["formed_path"]
    indices, y, x = zip(*[(index, n[0], n[1]) for index, n in enumerate(formed_path)])
    dataframes["FormedPath"] = pd.DataFrame({
        'path_num': indices,
        'y': y,
        'x': x
    })[dataframes["FormedPath"].columns]