import networkx as nx

import helpers.mazify.temp_options as options


def shortest_path(G: nx.Graph, source, target, weight, method: str = "dijkstra", test_only=False):
    """
    Find the shortest path between two nodes in a graph with optional edge weight updates.
    
    This function extends networkx's shortest_path with additional functionality to update edge weights
    after finding a path (scorched earth strategy).
    
    Args:
        G: The NetworkX graph.
        source: Starting node.
        target: Ending node.
        weight: Edge data key corresponding to the edge weight.
        method: Algorithm to use for shortest path. Defaults to "dijkstra".
        test_only: If True, don't update edge weights. Defaults to False.
        
    Returns:
        list: A list of nodes representing the shortest path from source to target, or None if no path exists.
        
    Note:
        If options.scorched_earth is True and test_only is False, the weights of the edges 
        in the found path will be multiplied by options.scorched_earth_weight_multiplier.
    """
    try:
        path = nx.shortest_path(G, source, target, weight=weight, method=method)

        # Update graph if scorched earth
        if options.scorched_earth and not test_only:
            for n in range(len(path) - 1):
                node_from, node_to = path[n], path[n + 1]
                G.edges[node_from, node_to]['weight'] = \
                    options.scorched_earth_weight_multiplier * G.edges[node_from, node_to]['weight']

        return path
    except nx.NetworkXNoPath:
        return None


def burn_path(G: nx.Graph, path):
    """
    Update edge weights along a given path in the graph.
    
    This is typically used to make certain paths less desirable for future route calculations.
    
    Args:
        G: The NetworkX graph.
        path: A list of nodes representing a path in the graph.
        
    Note:
        Only takes effect if options.scorched_earth is True. Multiplies the weight of each edge 
        in the path by options.scorched_earth_weight_multiplier.
    """
    # Update graph if scorched earth
    if options.scorched_earth:
        for n in range(len(path) - 1):
            node_from, node_to = path[n], path[n + 1]
            G.edges[node_from, node_to]['weight'] = \
                options.scorched_earth_weight_multiplier * G.edges[node_from, node_to]['weight']
