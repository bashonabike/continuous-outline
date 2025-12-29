import networkx as nx

import helpers.mazify.temp_options as options

def shortest_path(G: nx.Graph, source, target, weight, method: str = "dijkstra", test_only=False):
    try:
        path = nx.shortest_path(G, source, target, weight=weight, method=method)

        #Update graph if scorched earth
        if options.scorched_earth and not test_only:
            for n in range(len(path) - 1):
                node_from, node_to = path[n], path[n + 1]
                G.edges[node_from, node_to]['weight'] =\
                    options.scorched_earth_weight_multiplier*G.edges[node_from, node_to]['weight']

        return path
    except nx.NetworkXNoPath:
        return None

def burn_path(G: nx.Graph, path):
    # Update graph if scorched earth
    if options.scorched_earth:
        for n in range(len(path) - 1):
            node_from, node_to = path[n], path[n + 1]
            G.edges[node_from, node_to]['weight'] = \
                options.scorched_earth_weight_multiplier * G.edges[node_from, node_to]['weight']