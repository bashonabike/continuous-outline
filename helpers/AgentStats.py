import numpy as np


class AgentStats:
    def __init__(self):
        self.accum_defl_rad = 0.0
        self.oblit_mask_size = 0
        self.crossovers = 0
        self.length_of_connectors = 0 #NOTE: round off each to int precision
        #loopback amt?

        self.final_score = 0

    def set_final_score(self):
        self.final_score = self.oblit_mask_size/(self.accum_defl_rad + self.crossovers + self.length_of_connectors)