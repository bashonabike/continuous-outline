import helpers.Enums as Enums

class TourNode:
    def __init__(self, x, y, set:Enums.NodeSet):
        """
        Initializes a TourNode object.

        Args:
            x (int): The x-coordinate of the node.
            y (int): The y-coordinate of the node.
            set (Enums.NodeSet): The set of the node (OUTER or DETAIL).

        Attributes:
            x (int): The x-coordinate of the node.
            y (int): The y-coordinate of the node.
            set (Enums.NodeSet): The set of the node (OUTER or DETAIL).
            prev (TourNode): The previous node in the path.
            prev_angle_rad (float): The angle of the path from the previous node to this node.
            next (TourNode): The next node in the path.
            next_angle_rad (float): The angle of the path from this node to the next node.
            starter (bool): Whether this node is a starter node.
            ender (bool): Whether this node is an ender node.
            nodes_in_rings (list): A list of nodes that are in the current node's rings.
            nodes_in_rings_to_oblate (list): A list of nodes that are in the current node's rings and are not oblated.
            crowded (bool): Whether this node is crowded.
        """
        self.x = x
        self.y = y
        self.set = set
        self.prev = None
        self.prev_angle_rad = 0
        self.next = None
        self.next_angle_rad = 0
        self.starter = False
        self.ender = False
        self.nodes_in_rings = []
        self.nodes_in_rings_to_oblate = []
        self.crowded = False
