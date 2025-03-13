import numpy as np
import pandas as pd
import helpers.mazify.temp_options as options


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

def build_level_2_data(dataframes:dict, in_data:dict, out_data:dict):
    """
    Build the level 2 data for the sections and agent.
    :param outer_contours: dict of dataframes to set
    :param in_data: dict of objects inputted from previous level
    :param out_data: dict of objects to output
    :return: None
    """

    #Retrieve section header data
    section_header_df = dataframes["Sections"]
    height, width = section_header_df['height'][0], section_header_df['width'][0]
    shape = (height, width)
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
    #TODO: set indices group and join on indices
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
        x_coords = group_df['x_sec'].to_numpy()
        focus_sections.append(list(zip(y_coords, x_coords)))


    #Build section details (except adding nodes)
    from helpers.mazify.MazeSections import MazeSections, MazeSection, MazeSectionTracker
    sections_df = dataframes["Section"]
    sections_with_focus_df = pd.merge(sections_df, focus_sections_df, on=['y_sec', 'x_sec'], how='left')
    sections_with_focus_df.sort_values(['y_sec', 'x_sec', 'focus_mask'], inplace=True)
    sections_nd = np.zeros(shape, dtype=MazeSection)
    for i, group_df in sections_with_focus_df.groupby(['y_sec', 'x_sec']):
        bounds =  (group_df['y_start'], group_df['y_end'], group_df['x_start'], group_df['x_end'])
        focus_masks_l = group_df['focus_mask'].to_list() if (group_df['focus_mask'] is not None
                                                            and not pd.isna(group_df['focus_mask'])) else []
        cur_section = MazeSection.from_df(bounds, group_df['num_edge_pixels'], group_df['y_sec'], group_df['x_sec'],
                                          group_df['num_edge_pixels'], focus_masks_l, group_df['dumb_req'],
                                          group_df['dumb_opt'])
        sections_nd[group_df['y_sec'], group_df['x_sec']] = cur_section


    #Build edge path, trackers, nodes
    from helpers.mazify.EdgePath import EdgePath
    from helpers.mazify.EdgeNode import EdgeNode
    edge_paths_df, edge_nodes_df, trackers_df = dataframes["EdgePath"], dataframes["PathNode"], dataframes["SectionTracker"]
    edge_paths_all_df = pd.merge(edge_paths_df, trackers_df, on=['path'], how='inner')
    edge_paths_all_df = pd.merge(edge_paths_all_df, edge_nodes_df, on=['path', 'tracker_num'], how='inner')
    edge_paths_df.sort_values(['path', 'tracker_num', 'node'], inplace=True)
    edge_paths, cur_trackers, cur_path, cur_tracker_path = [], [], [], []
    cur_path_num, cur_outer, cur_closed, cur_weight = -1, False, False, 0
    cur_y_sec, cur_x_sec = -1, -1
    cur_tracker_num = -1
    for (path_num, tracker_num), path_tracker_df in edge_paths_df.groupby(['path', 'tracker_num']):
        if cur_tracker_num != path_tracker_df['tracker_num']:
            prev_tracker = cur_trackers[-1] if len(cur_trackers) > 0 else None
            cur_trackers.append(MazeSectionTracker.from_df(sections_nd[cur_y_sec, cur_x_sec], cur_tracker_path[0],
                                                           cur_tracker_path[-1], cur_tracker_num, prev_tracker,
                                                           cur_tracker_path))

            for node in cur_tracker_path:
                node.section_tracker = cur_trackers[-1]
                node.section_tracker_num = cur_tracker_num

            cur_tracker_num, cur_tracker_path = path_tracker_df.at[0, 'tracker_num'], []
            cur_y_sec, cur_x_sec = path_tracker_df.at[0, 'y_sec'], path_tracker_df.at[0, 'x_sec']

        if cur_path_num != path_tracker_df.at[0, 'path']:
            if cur_path_num != -1:
                edge_paths.append(EdgePath.from_df(cur_path_num, cur_outer, cur_closed, cur_weight, cur_path,
                                                   cur_trackers))

            for node in cur_path:
                node.path = edge_paths[-1]

            cur_trackers, cur_path = [], []
            cur_path_num, cur_outer, cur_closed, cur_weight = (path_tracker_df.at[0, 'path'],
                                                                path_tracker_df.at[0, 'is_outer'],
                                                                path_tracker_df.at[0, 'is_closed'],
                                                                path_tracker_df.at[0, 'custom_weight'])

        cur_node = EdgeNode.from_df(path_tracker_df.at[0, 'y'], path_tracker_df.at[0, 'x'], path_tracker_df.at[0, 'path'],
                                    path_tracker_df.at[0, 'node'], path_tracker_df.at[0, 'is_outer'],
                                    sections_nd[cur_y_sec, cur_x_sec])
        cur_path.append(cur_node)
        cur_tracker_path.append(cur_node)

    #Final push last edge path
    last_next_tracker = cur_trackers[0] if cur_closed else None
    prev_tracker = cur_trackers[-1] if len(cur_trackers) > 0 else None
    cur_trackers.append(MazeSectionTracker.from_df(sections_nd[cur_y_sec, cur_x_sec], cur_tracker_path[0],
                                                   cur_tracker_path[-1], cur_tracker_num, prev_tracker,
                                                   cur_tracker_path, last_next_tracker))
    for node in cur_tracker_path:
        node.section_tracker = cur_trackers[-1]
        node.section_tracker_num = cur_tracker_num

    edge_paths.append(EdgePath.from_df(cur_path_num, cur_outer, cur_closed, cur_weight, cur_path,
                                       cur_trackers))


    #Configure sections object
    sections_object = MazeSections.from_df(options.maze_sections_across, options.maze_sections_across, focus_sections,
                                           sections_nd, y_grade, x_grade, path_graph)
    out_data['sections'] = sections_object


    #Configure agent object
    from helpers.mazify.MazeAgent import MazeAgent
    agent_df = dataframes["Agent"]
    start_node = edge_paths[agent_df.at[0, 'start_path'] - 1].path[agent_df.at[0, 'start_node']]
    end_node = edge_paths[agent_df.at[0, 'end_path'] - 1].path[agent_df.at[0, 'end_node']]
    agent = MazeAgent.from_df(in_data['outer_edges'], in_data['outer_contours'], in_data['inner_edges'], in_data['inner_contours'],
                              sections_object, edge_paths, agent_df.at[0, 'max_tracker_size'], start_node, end_node)
    out_data['agent'] = agent
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