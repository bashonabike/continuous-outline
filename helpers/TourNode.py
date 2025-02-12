import helpers.Enums as Enums

class TourNode:
    def __init__(self, x, y, set:Enums.NodeSet):
        self.x = x
        self.y = y
        self.set = set
        self.prev = None
        self.prev_angle_rad = 0
        self.next = None
        self.next_angle_rad = 0
        self.starter = False
        self.ender = False