
import helpers.DECREPIT.old_method.ParsePathsIntoObjects as parse
import helpers.Enums as enums

class NodeSet:
    def __init__(self, nodes, dims):
        """
        Initializes a NodeSet object from a list of nodes and the dimensions of the grid.

        Args:
            nodes (list): A list of Node objects.
            dims (tuple): A tuple of two integers, representing the dimensions of the grid.

        Attributes:
            all_nodes (list): A list of all Node objects.
            x_sorted_nodes (list): A list of Node objects sorted by their x-coordinate.
            y_sorted_nodes (list): A list of Node objects sorted by their y-coordinate.
            gridded_nodes (numpy array): A 2D array of Node objects, where each element is a Node object if it exists at that position in the grid, otherwise None.
            outer_nodes (list): A list of Node objects that are OUTER nodes.
            detail_nodes (list): A list of Node objects that are DETAIL nodes.
            outer_bookends (list): A list of Node objects that are OUTER nodes and are either starters or enders.
            bookends (list): A list of Node objects that are either starters or enders.
            oblit_nodes (list): A list of Node objects that have been oblit.

        """
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
        """
        Resets all nodes to their original state.

        This method is used to reset all nodes after a tour has been completed.

        Args:
            None

        Returns:
            None
        """
        for node in self.all_nodes:
            if node.set == enums.NodeSet.OUTEROBLIT:
                node.set = enums.NodeSet.OUTER
            elif  node.set == enums.NodeSet.DETAILOBLIT:
                node.set = enums.NodeSet.DETAIL

            node.crowded = False