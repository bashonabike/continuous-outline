import numpy as np
import pandas as pd
import helpers.mazify.temp_options as options
import time


def build_level_1_data(dataframes:dict, out_data:dict):
    """
    Build the level 1 data for the contours and edges.
    :param outer_contours: dict of dataframes to set
    :param input_data: dict of objects to input
    :return: None
    """
    import cv2

    def pixel_map_from_edge_contours(shape, contours, offset_idx):
        contours_for_cv = [np.array([[[p[1], p[0]]] for p in contour]) for contour in contours]
        edges_final = np.zeros(shape, dtype=np.uint16)
        for contour_idx in range(len(contours_for_cv)):
            cv2.drawContours(edges_final, contours_for_cv, contour_idx,
                             (contour_idx + 1 + offset_idx, contour_idx + 1 + offset_idx,
                              contour_idx + 1 + offset_idx))
        return edges_final

    #Set contours lists
    contours_hdr_df, contours_det_df = dataframes["Contours"], dataframes["Contour"]
    outer_last_idx = int(contours_hdr_df[contours_hdr_df['is_outer'] == 1]['contour'].max())
    contours_all_df = pd.merge(contours_hdr_df, contours_det_df, on='contour', how='inner')
    contours_all_df.sort_values(['contour', 'point_num'], inplace=True)
    outer_contours, inner_contours = [], []
    for (contour, is_outer, is_inner), contour_df in contours_all_df.groupby(['contour', 'is_outer', 'is_inner']):
        if is_outer:
            outer_contours.append(list(zip(contour_df['y'], contour_df['x'])))
        else:
            inner_contours.append(list(zip(contour_df['y'], contour_df['x'])))

    out_data["outer_contours"], out_data["inner_contours"] = outer_contours, inner_contours


    #Set edges pixel maps
    shape = contours_hdr_df["img_height"][0], contours_hdr_df["img_width"][0]
    outer_edges = pixel_map_from_edge_contours(shape, outer_contours, 0)
    inner_edges = pixel_map_from_edge_contours(shape, inner_contours, outer_last_idx)

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
    img_height, img_width = section_header_df['img_height'][0], section_header_df['img_width'][0]
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
    graph_edges_formed_df = pd.merge(graph_edges_df, graph_nodes_df, left_on='from_node', right_on='graph_node', how='inner')
    graph_edges_formed_df = pd.merge(graph_edges_formed_df, graph_nodes_df, left_on='to_node', right_on='graph_node',
                                     how='inner', suffixes=('__from', '__to')
                                     )
    blank_from = graph_edges_formed_df['graph_node_path__from'] == 0
    blank_to = graph_edges_formed_df['graph_node_path__to'] == 0
    blank_to_blank_df = graph_edges_formed_df[blank_from & blank_to]
    blank_to_edge_df = graph_edges_formed_df[blank_from & (~blank_to)]
    edge_to_blank_df = graph_edges_formed_df[(~blank_from) & blank_to]
    edge_to_edge_df = graph_edges_formed_df[~(blank_from | blank_to)]

    for tuple_len, df in zip([(2, 2), (2, 4), (4, 2), (4, 4)],
                             [blank_to_blank_df, blank_to_edge_df, edge_to_blank_df, edge_to_edge_df]):
        nodes_set = []
        for i, dir in enumerate(["from", "to"]):
            if tuple_len[i] == 4:
                nodes_set.append(list(zip(
                    df['graph_node_y__' + dir].to_numpy(),
                    df['graph_node_x__' + dir].to_numpy(),
                    df['graph_node_path__' + dir].to_numpy(),
                    df['graph_node_tracker__' + dir].to_numpy()
                )))
            else:
                nodes_set.append(list(zip(
                    df['graph_node_y__' + dir].to_numpy(),
                    df['graph_node_x__' + dir].to_numpy()
                )))

        weights = graph_edges_df['weight'].to_numpy()

        edges_to_add = list(zip(nodes_set[0], nodes_set[1], ({'weight': w} for w in weights)))
        path_graph.add_edges_from(edges_to_add)


    #Build focus region sections
    focus_sections_df = dataframes["FocusSection"]
    focus_sections_df.sort_values(['focus_mask'], inplace=True)
    focus_sections = []
    for i, group_df in focus_sections_df.groupby(['focus_mask']):
        y_coords = group_df['y_sec'].to_numpy()
        x_coords = group_df['x_sec'].to_numpy()
        focus_sections.append(list(zip(y_coords, x_coords)))


    #Build section details (except adding nodes)
    from helpers.mazify.MazeSections import MazeSections, MazeSection, MazeSectionTracker
    sections_df = dataframes["Section"]
    sections_nd = np.zeros(shape, dtype=MazeSection)

    start = time.time_ns()
    sections_dicts = sections_df.to_dict('records')
    maze_sections_flat_l = [MazeSection.from_df(**row) for row in sections_dicts]

    # Pre-allocate lists for y_sec, x_sec, and sections
    y_coords = np.array([sec.y_sec for sec in maze_sections_flat_l])
    x_coords = np.array([sec.x_sec for sec in maze_sections_flat_l])
    sections_nd[y_coords, x_coords] = np.array(maze_sections_flat_l)
    end = time.time_ns()
    print(str((end - start) / 1e6) + " ms to do NEW sections")

    # try:
    start = time.time_ns()
    #Build edge path, trackers, nodes
    from helpers.mazify.EdgePath import EdgePath
    from helpers.mazify.EdgeNode import EdgeNode
    edge_paths_df, edge_nodes_df, trackers_df = dataframes["EdgePath"], dataframes["PathNode"], dataframes["SectionTracker"]
    edge_paths_all_df = pd.merge(edge_paths_df, trackers_df, on=['path'], how='inner')
    edge_paths_all_df = pd.merge(edge_paths_all_df, edge_nodes_df, on=['path', 'tracker_num'], how='inner', suffixes=("", "_tracker"))
    edge_paths_all_df.sort_values(['path', 'tracker_num', 'node'], inplace=True)
    edge_paths_all_df['section'] = sections_nd[edge_paths_all_df['y_sec'], edge_paths_all_df['x_sec']]

    #Build all stub trackers
    edge_paths_all_df.set_index(['path', 'tracker_num'], inplace=True)
    edge_paths_all_df['path_num'] = edge_paths_all_df.index.get_level_values(0)
    trackers_df = edge_paths_all_df.groupby(level=[0, 1]).agg(
        num_nodes=('node', 'count'),
        section=('section', 'first')

    )
    trackers_df['path_num'] = trackers_df.index.get_level_values(0)
    trackers_df['tracker_num'] = trackers_df.index.get_level_values(1)
    all_trackers_l = [MazeSectionTracker.from_df(**row) for row in trackers_df.to_dict('records')]



    # Build all stub paths
    paths_df = edge_paths_all_df.groupby(level=0).agg(
        num_nodes=('node', 'count'),
        is_outer=('is_outer', 'first'),
        is_closed=('is_closed', 'first'),
        custom_weight=('custom_weight', 'first')
    )
    paths_df['path_num'] = paths_df.index.get_level_values(0)
    all_paths_l = [EdgePath.from_df(**row) for row in paths_df.to_dict('records')]

    #Build all nodes
    edge_paths_all_df['path_object'] = np.array(all_paths_l)[edge_paths_all_df['path_num'] - 1]
    edge_paths_all_df['tracker_object'] = np.array(all_trackers_l)[edge_paths_all_df['path_tracker_seq']]
    nodes_setting_df = edge_paths_all_df[['y', 'x', 'path_num', 'node', 'is_outer', 'section', 'path_object',
                                          'tracker_object']]
    all_nodes_l = [EdgeNode.from_df(**row) for row in nodes_setting_df.to_dict('records')]

    #Set into trackers and paths
    path_idx, path_tracker_idx, path_nodes_idx, tracker_nodes_idx = 0, 0, 0, 0
    for i in range(len(all_trackers_l)):
        if path_idx < len(all_paths_l) and all_trackers_l[i].path_num != path_idx + 1:
            path_nodes = all_nodes_l[path_nodes_idx:path_nodes_idx + all_paths_l[path_idx].from_db_num_nodes]
            all_paths_l[path_idx].set_path_and_trackers(path_nodes, all_trackers_l[path_tracker_idx:i])
            if all_paths_l[path_idx].closed:
                all_trackers_l[path_tracker_idx].prev_tracker = all_trackers_l[i - 1]
                all_trackers_l[i - 1].next_tracker = all_trackers_l[path_tracker_idx]
            path_nodes_idx += all_paths_l[path_idx].from_db_num_nodes
            path_idx += 1
            path_tracker_idx = i

        if all_trackers_l[i].tracker_num == 0:
            prev_tracker, next_tracker = None, None if i >= len(all_trackers_l) else all_trackers_l[i + 1]
        elif i >= len(all_trackers_l) - 1 or all_trackers_l[i + 1].tracker_num == 0:
            prev_tracker, next_tracker = all_trackers_l[i - 1], None
        else:
            prev_tracker, next_tracker = all_trackers_l[i - 1], all_trackers_l[i + 1]
        all_trackers_l[i].set_nodes_and_neighbours(all_nodes_l[tracker_nodes_idx:tracker_nodes_idx +
                                                                                 all_trackers_l[i].from_db_num_nodes],
                                                   prev_tracker, next_tracker)
        tracker_nodes_idx += all_trackers_l[i].from_db_num_nodes

    #Set final path
    path_nodes = all_nodes_l[path_nodes_idx:path_nodes_idx + all_paths_l[path_idx].from_db_num_nodes]
    all_paths_l[path_idx].set_path_and_trackers(path_nodes, all_trackers_l[path_tracker_idx:])
    if all_paths_l[path_idx].closed:
        all_trackers_l[path_tracker_idx].prev_tracker = all_trackers_l[len(all_trackers_l) - 1]
        all_trackers_l[len(all_trackers_l) - 1].next_tracker = all_trackers_l[path_tracker_idx]

    end = time.time_ns()
    print(str((end - start) / 1e6) + " ms to do path trackers nodes")


    #Configure sections object
    sections_object = MazeSections.from_df(options.maze_sections_across, options.maze_sections_across, focus_sections,
                                           sections_nd, y_grade, x_grade, img_height, img_width, path_graph)
    out_data['sections'] = sections_object


    #Configure agent object
    from helpers.mazify.MazeAgent import MazeAgent
    agent_df = dataframes["Agent"]
    start_node = all_paths_l[agent_df.at[0, 'start_path'] - 1].path[agent_df.at[0, 'start_node']]
    end_node = all_paths_l[agent_df.at[0, 'end_path'] - 1].path[agent_df.at[0, 'end_node']]
    agent = MazeAgent.from_df(in_data['outer_edges'], in_data['outer_contours'], in_data['inner_edges'], in_data['inner_contours'],
                              sections_object, all_paths_l, agent_df.at[0, 'max_tracker_size'], start_node, end_node)
    out_data['agent'] = agent
def build_level_3_data(dataframes:dict, out_data:dict):
    """
    Set the level 2 data for the contours and edges.
    :param outer_contours: dict of dataframes to set
    :param out_data: dict of objects to output
    :return: None
    """

    #Retrieve raw path data
    # dataframes["RawPath"].set_index('path_num', inplace=True)
    out_data["raw_path"] = list(zip(dataframes["RawPath"]['y'], dataframes["RawPath"]['x']))

def build_level_4_data(dataframes:dict, out_data:dict):
    """
    Set the level 2 data for the contours and edges.
    :param outer_contours: dict of dataframes to set
    :param out_data: dict of objects to output
    :return: None
    """

    #Retrieve raw path data
    # dataframes["FormedPath"].set_index('path_num', inplace=True)
    out_data["formed_path"] = list(zip(dataframes["FormedPath"]['y'], dataframes["FormedPath"]['x']))