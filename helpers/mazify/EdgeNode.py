class EdgeNode:
    def __init__(self, y, x, path, rev_dir, rev_dir_smoothed, fwd_dir, fwd_dir_smoothed,  fwd_displ, is_outer):
        self.y, self.x = y, x
        self.path, self.path_num = path, path.num
        self.point = (y, x)
        self.rev_dir, self.rev_dir_smoothed = rev_dir, rev_dir_smoothed
        self.fwd_dir, self.fwd_dir_smoothed = fwd_dir, fwd_dir_smoothed
        self.fwd_displ = fwd_displ
        self.filled = False
        self.outer = is_outer
        self.section = None
        self.section_tracker = None
        self.section_tracker_num = -1

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


