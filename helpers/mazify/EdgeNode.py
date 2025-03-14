class EdgeNode:
    def __init__(self, y, x, path, node_num, is_outer, from_df=False, section=None, path_num=None,
                 section_tracker=None):
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
        return cls(y, x, path_object, node, is_outer, from_df=True, section=section, path_num=path_num,
                   section_tracker=tracker_object)

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


