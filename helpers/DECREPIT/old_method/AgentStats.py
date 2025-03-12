import numpy as np


class AgentStats:
    def __init__(self):
        self.accum_defl_rad = 0.0
        self.oblit_mask_size = 0
        self.oblit_mask_outer_size = 0
        self.crowding = 0
        self.length_of_connectors = 0 #NOTE: round off each to int precision
        #loopback amt?

        self.pathtime = 0.0
        self.searchtime = 0.0
        self.oblatedrawtime = 0.0
        self.oblatemaskime = 0.0
        self.oblatefindpixelstime = 0.0
        self.oblateiternodestime = 0.0

        self.final_score = 0

    def set_final_score(self):
        self.final_score = (0.05*self.oblit_mask_size + self.oblit_mask_outer_size +
                            (10000000000.0/(self.accum_defl_rad + 200*self.crowding + self.length_of_connectors + 1)))