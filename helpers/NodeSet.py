
import helpers.ParsePathsIntoObjects as parse
import helpers.Enums as enums

class NodeSet:
    def __init__(self, nodes, dims):
        self.all_nodes = nodes

        self.x_sorted_nodes = sorted(self.all_nodes, key=lambda n: (n.x, n.y))
        self.y_sorted_nodes = sorted(self.all_nodes, key=lambda n: (n.y, n.x))
        self.gridded_nodes = parse.grid_nodes(self.x_sorted_nodes, dims[0], dims[1])
        self.outer_nodes = [n for n in self.all_nodes if n.set == enums.NodeSet.OUTER]
        self.detail_nodes = [n for n in self.all_nodes if n.set == enums.NodeSet.DETAIL]
        self.outer_bookends = [n for n in self.all_nodes if n.set == enums.NodeSet.OUTER and
                               (n.starter or n.ender)]
        self.bookends = [n for n in self.all_nodes if n.starter or n.ender]
        self.oblit_nodes = []
        parse.search_for_cand_next_nodes(self.bookends, self.gridded_nodes)

    def reset_oblit(self):
        for node in self.all_nodes:
            if node.set == enums.NodeSet.OUTEROBLIT:
                node.set = enums.NodeSet.OUTER
            elif  node.set == enums.NodeSet.DETAILOBLIT:
                node.set = enums.NodeSet.DETAIL

            node.crowded = False