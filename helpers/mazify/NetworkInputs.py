class NetworkInputs:
    def __init__(self):
        self.inputs = []

    def add_input(self, compass_type, compass_dir, on_edge):
        self.inputs.append(NetworkInput(compass_type, compass_dir, on_edge))

    def find_input(self, compass_type, compass_dir):
        for input in self.inputs:
            if input.compass_type == compass_type and input.compass_dir == compass_dir:
                return input

class NetworkInput:
    def __init__(self, compass_type, compass_dir, on_edge):
        self.compass_type = compass_type
        self.compass_dir = compass_dir
        self.on_edge = on_edge
        self.value = 0.0

    def set_value(self, value):
        self.value = value