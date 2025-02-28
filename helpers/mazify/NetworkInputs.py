class NetworkInputs:
    def __init__(self):
        self.inputs = []

    def add_input(self, parent_compass, compass_dir):
        self.inputs.append(NetworkInput(parent_compass, compass_dir))

    def find_input(self, parent_compass, compass_dir):
        for input in self.inputs:
            if input.parent_compass == parent_compass and input.compass_dir == compass_dir:
                return input

class NetworkInput:
    def __init__(self, parent_compass, compass_dir):
        self.parent_compass = parent_compass
        self.compass_dir = compass_dir
        self.value = 0.0

    def set_value(self, value):
        self.value = value