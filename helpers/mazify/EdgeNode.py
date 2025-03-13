class EdgeNode:
    def __init__(self, y, x, path, node_num, is_outer, hash=0):
        self.y, self.x = y, x
        self.path, self.path_num = path, path.num
        self.num = node_num
        self.point = (y, x)
        self.outer = is_outer
        self.section = None
        self.section_tracker = None
        self.section_tracker_num = -1
        # self.hash = hash

    def set_section(self, section, section_tracker):
        self.section = section
        self.section_tracker = section_tracker
        self.section_tracker_num = section_tracker.tracker_num

    def set_prev_node(self, node):
        self.prev_node = node

    def set_next_node(self, node):
        self.next_node = node

    def walk(self, reverse:bool=False):
        if not reverse:
            return self.next_node
        else:
            return self.prev_node


