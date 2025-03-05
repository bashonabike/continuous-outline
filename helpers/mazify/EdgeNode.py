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
        self.section_tracker_num = -1

    def set_section(self, section, section_tracker_num):
        self.section = section
        self.section_tracker_num = section_tracker_num


