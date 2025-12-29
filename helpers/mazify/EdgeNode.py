class EdgeNode:
    """
    Represents a node in a graph that forms the edges of a maze or path structure.
    
    This class is used to manage the connections between different segments of a path
    and track their positions and relationships within the larger structure.
    """

    def __init__(self, y, x, path, node_num, is_outer, from_df=False, section=None, path_num=None,
                 section_tracker=None):
        """
        Initialize an EdgeNode instance.
        
        Args:
            y (int): The y-coordinate of the node.
            x (int): The x-coordinate of the node.
            path: The path object this node belongs to.
            node_num (int): The unique identifier for this node.
            is_outer (bool): Whether this node is on the outer boundary.
            from_df (bool, optional): Whether this node is being created from a DataFrame. Defaults to False.
            section: The section of the path this node belongs to. Defaults to None.
            path_num (int, optional): The path number. Defaults to None.
            section_tracker: Object to track section information. Defaults to None.
        """
        self.y, self.x = y, x
        self.path, self.path_num = path, path.num
        self.num = node_num
        self.point = (y, x)
        self.outer = is_outer
        self.section = section
        if not from_df:
            self.section_tracker = None
            self.section_tracker_num = -1
        else:
            self.section_tracker = section_tracker
            self.section_tracker_num = section_tracker.tracker_num

    @classmethod
    def from_df(cls, y, x, path_num, node, is_outer, section, path_object, tracker_object):
        """
        Create an EdgeNode instance from DataFrame data.
        
        Args:
            y (int): The y-coordinate of the node.
            x (int): The x-coordinate of the node.
            path_num (int): The path number.
            node: The node identifier.
            is_outer (bool): Whether this node is on the outer boundary.
            section: The section of the path this node belongs to.
            path_object: The path object this node belongs to.
            tracker_object: The section tracker object.
            
        Returns:
            EdgeNode: A new EdgeNode instance created from the provided DataFrame data.
        """
        return cls(y, x, path_object, node, is_outer, from_df=True, section=section, path_num=path_num,
                   section_tracker=tracker_object)

    def set_section(self, section, section_tracker):
        """
        Set the section and section tracker for this node.
        
        Args:
            section: The section of the path this node belongs to.
            section_tracker: The section tracker object to associate with this node.
        """
        self.section = section
        self.section_tracker = section_tracker
        self.section_tracker_num = section_tracker.tracker_num

    def set_prev_node(self, node):
        """
        Set the previous node in the path.
        
        Args:
            node (EdgeNode): The previous node in the path.
        """
        self.prev_node = node

    def set_next_node(self, node):
        """
        Set the next node in the path.
        
        Args:
            node (EdgeNode): The next node in the path.
        """
        self.next_node = node

    def walk(self, reverse: bool = False):
        """
        Get the next or previous node in the path.
        
        Args:
            reverse (bool, optional): If True, return the previous node; otherwise, return the next node.
                                    Defaults to False.
            
        Returns:
            EdgeNode: The next or previous node in the path, depending on the reverse parameter.
        """
        if not reverse:
            return self.next_node
        else:
            return self.prev_node
