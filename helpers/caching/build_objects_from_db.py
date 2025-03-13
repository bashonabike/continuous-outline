import numpy as np
import pandas as pd


def build_level_1_data(dataframes:dict, out_data:dict):
    """
    Build the level 1 data for the contours and edges.
    :param outer_contours: dict of dataframes to set
    :param input_data: dict of objects to input
    :return: None
    """
    #Set contours lists
    contours_hdr_df, contours_det_df = dataframes["Contours"], dataframes["Contour"]
    contours_all_df = pd.merge(contours_hdr_df, contours_det_df, on='contour', how='inner')
    contours_all_df.sort_values(['contour', 'path_num'], inplace=True)
    outer_contours, inner_contours = [], []
    for (contour, is_outer, is_inner), contour_df in contours_all_df.groupby(['contour', 'is_outer', 'is_inner']):
        if is_outer:
            outer_contours.append(list(zip(contour_df['y'], contour_df['x'])))
        else:
            inner_contours.append(list(zip(contour_df['y'], contour_df['x'])))

    out_data["outer_contours"], out_data["inner_contours"] = outer_contours, inner_contours


    #Set edges pixel maps
    edges_pixel_map_df = dataframes["EdgesPixelMap"]
    shape = (edges_pixel_map_df['y'].max() + 1, edges_pixel_map_df['x'].max() + 1)
    outer_edges, inner_edges = np.zeros(shape, dtype=np.uint16), np.zeros(shape, dtype=np.uint16)

    y_coords = edges_pixel_map_df['y'].to_numpy()
    x_coords = edges_pixel_map_df['x'].to_numpy()
    outer_values = edges_pixel_map_df['outer_edge_num'].to_numpy()
    inner_values = edges_pixel_map_df['inner_edge_num'].to_numpy()

    outer_edges[y_coords, x_coords] = outer_values
    inner_edges[y_coords, x_coords] = inner_values

    out_data["outer_edges"], out_data["inner_edges"] = outer_edges, inner_edges


    #Set focus masks
    focus_masks_df = dataframes["FocusMask"]
    focus_masks_df.sort_values(['focus_mask'], inplace=True)
    focus_masks = []
    for focus_mask_value, group_df in focus_masks_df.groupby('focus_mask'):
        mask_array = np.zeros(shape, dtype=bool)
        y_coords = group_df['y'].to_numpy()
        x_coords = group_df['x'].to_numpy()
        mask_array[y_coords, x_coords] = True
        focus_masks.append(mask_array)

    out_data["focus_masks"] = focus_masks

def build_level_2_data(dataframes:dict, out_data:dict):
    """
    Build the level 2 data for the sections and agent.
    :param outer_contours: dict of dataframes to set
    :param out_data: dict of objects to output
    :return: None
    """

    #Retrieve section header data
    section_header_df = dataframes["Sections"]
    height, width = section_header_df['height'][0], section_header_df['width'][0]
    y_grade, x_grade = section_header_df['y_grade'][0], section_header_df['x_grade'][0]

    #Set up graph
    import networkx as nx
    path_graph = nx.Graph()

    graph_nodes_df = dataframes["GraphNode"]
    blank_nodes_df = graph_nodes_df[graph_nodes_df['graph_node_path'] == 0]
    edged_nodes_df = graph_nodes_df[graph_nodes_df['graph_node_path'] != 0]

    # Add nodes where graph_node_path is 0
    if not blank_nodes_df.empty:
        nodes_0 = list(zip(
            zip(blank_nodes_df['graph_node_y'], blank_nodes_df['graph_node_x']),
            ({'category': cat} for cat in blank_nodes_df['category'])
        ))
        path_graph.add_nodes_from(nodes_0)

    # Add nodes where graph_node_path is not 0
    if not edged_nodes_df.empty:
        nodes_non_0 = list(zip(
            zip(edged_nodes_df['graph_node_y'], edged_nodes_df['graph_node_x'],
                edged_nodes_df['graph_node_path'], edged_nodes_df['graph_node_tracker']),
            ({'category': cat} for cat in edged_nodes_df['category'])
        ))
        path_graph.add_nodes_from(nodes_non_0)

    # Set edges
    graph_edges_df = dataframes["GraphEdge"]
    graph_edges_formed_df = pd.merge(graph_edges_df, graph_nodes_df, left_on='from_node', right_on='node', how='inner')
    graph_edges_formed_df = pd.merge(graph_edges_formed_df, graph_nodes_df, left_on='to_node', right_on='node',
                                     how='inner', suffixes=('_from', '_to'))
    blank_to_blank_df = graph_edges_formed_df[graph_edges_formed_df['graph_node_path_from'] == 0 and
                                           graph_edges_formed_df['graph_node_path_to'] == 0]
    blank_to_edge_df = graph_edges_formed_df[graph_edges_formed_df['graph_node_path_from'] == 0 and
                                           graph_edges_formed_df['graph_node_path_to'] > 0]
    edge_to_blank_df = graph_edges_formed_df[graph_edges_formed_df['graph_node_path_from'] > 0 and
                                           graph_edges_formed_df['graph_node_path_to'] == 0]
    edge_to_edge_df = graph_edges_formed_df[graph_edges_formed_df['graph_node_path_from'] > 0 and
                                           graph_edges_formed_df['graph_node_path_to'] > 0]

    for tuple_len, df in zip([(2, 2), (2, 4), (4, 2), (4, 4)],
                             [blank_to_blank_df, blank_to_edge_df, edge_to_blank_df, edge_to_edge_df]):
        nodes_set = []
        for i, dir in enumerate(["from", "to"]):
            if tuple_len[i] == 4:
                nodes_set.append(list(zip(
                    graph_edges_df['graph_node_y_' + dir].to_numpy(),
                    graph_edges_df['graph_node_x_' + dir].to_numpy(),
                    graph_edges_df['graph_node_path_' + dir].to_numpy(),
                    graph_edges_df['graph_node_tracker_f' + dir].to_numpy()
                )))
            else:
                nodes_set.append(list(zip(
                    graph_edges_df['graph_node_y_' + dir].to_numpy(),
                    graph_edges_df['graph_node_x_' + dir].to_numpy()
                )))

        weights = graph_edges_df['weight'].to_numpy()

        edges_to_add = list(zip(nodes_set[0], nodes_set[1], ({'weight': w} for w in weights)))
        path_graph.add_edges_from(edges_to_add)


    #Build focus region sections
    focus_sections_df = dataframes["FocusSection"]
    focus_sections_df.sort_values(['focus_mask'], inplace=True)
    focus_sections = []
    for i, focus_mask_value, group_df in focus_sections_df.groupby(['focus_mask']):
        y_coords = group_df['y_sec'].to_numpy()
        x_coords =   group_df['x_sec'].to_numpy()
        focus_sections.append(list(zip(y_coords, x_coords)))


    #Set sections details
    y_coords, x_coords = np.indices(sections.sections.shape)  # Generate y and x coordinates using indices

    # Flatten the arrays
    y_flat = y_coords.flatten()
    x_flat = x_coords.flatten()
    sections_flat = sections.sections.flatten()
    dumb_req = np.array([section.dumb_req for section in sections_flat])
    dumb_opt = np.array([section.dumb_opt for section in sections_flat])

    #Create dataframe
    dataframes["Section"] = pd.DataFrame({
        'y_sec': y_flat,
        'x_sec': x_flat,
        'dumb_req': dumb_req,
        'dumb_opt': dumb_opt
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
            'path': path.num
        })[dataframes["PathNode"].columns]
        dataframes["PathNode"] = pd.concat([dataframes["PathNode"], partial_df], ignore_index=True)

    #Set section trackers
    max_tracker_size = 0
    for edge_path in edge_paths:
        trackers = edge_path.section_tracker
        if len(trackers) > max_tracker_size: max_tracker_size = len(trackers)
        in_node,out_node,path,rev_in_node,rev_out_node,tracker_num,prev_tracker,next_tracker,y_sec,x_sec = zip(*[(
            t.in_node,t.out_node,t.path_num,t.rev_in_node.num,t.rev_out_node.num,t.tracker_num,
            t.prev_tracker.num,t.next_tracker.num,t.section.y_sec,t.section.x_sec) for t in trackers])
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
            'x_sec': x_sec
        })[dataframes["SectionTracker"].columns]
        dataframes["SectionTracker"] = pd.concat([dataframes["SectionTracker"], partial_df], ignore_index=True)




    #Set into dataframe
    dataframes["GraphEdge"] = pd.DataFrame({
        'from_node': hash_in,
        'to_node': hash_out,
        'weight': weights
    })[dataframes["GraphEdge"].columns]


    #Set agent
    dataframes["Agent"] = pd.DataFrame({
        'start_node': agent.start_node.num,
        'end_node': agent.end_node.num,
        'start_path': agent.start_node.path_num,
        'end_path': agent.end_node.path_num
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
    dataframes["RawPath"] = pd.DataFrame({
        'path_num': indices,
        'y': y,
        'x': x
    })[dataframes["FormedPath"].columns]