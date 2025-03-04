class EdgeNode:
    def __init__(self, y, x, fwd_dir, fwd_dir_smoothed,  fwd_displ):
        self.y, self.x = y, x
        self.fwd_dir, self.fwd_dir_smoothed = fwd_dir, fwd_dir_smoothed
        self.fwd_displ = fwd_displ


