from helpers.mazify.NetworkInputs import NetworkInputs,NetworkInput
from helpers.Enums import CompassDir, CompassType
import helpers.mazify.NetworkInputs as inputs
import helpers.mazify.temp_options as options

import math

def get_direction(network_inputs: inputs.NetworkInputs, on_edge=True):
    #Compute net compass directions
    compass_net = {CompassDir.N: 0.0, CompassDir.E: 0.0, CompassDir.S: 0.0, CompassDir.W: 0.0}
    inner_draw = 0.0
    for input in network_inputs.inputs:
        if on_edge and not input.on_edge: continue
        match input.compass_type:
            case CompassType.legality_compass:
                weight = 20.0/2.4
            case CompassType.proximity_compass:
                weight = 2.0/2
            case CompassType.intersects_compass:
                weight = 1.0/2.4
            case CompassType.outer_attraction_compass:
                weight = 1.0/1000
            case CompassType.parallels_compass:
                # weight = 1.0
                weight = 1.0/20.0
            case CompassType.deflection_compass:
                weight = 2.0/2.4
            case CompassType.inner_attraction:
                weight = 1.0
            case _:
                weight = 0.0

        if input.compass_dir is not None:
            compass_net[input.compass_dir] += input.value * weight
        else:
            inner_draw += input.value * weight

        #TODO: configure inner_draw to increase weight of running parallels within current section

    #Determine strongest pull
    y, x = 0.0, 0.0
    if compass_net[CompassDir.N] > compass_net[CompassDir.S]:
        y = compass_net[CompassDir.N]
    else:
        y = (-1)*compass_net[CompassDir.S]

    if compass_net[CompassDir.E] > compass_net[CompassDir.W]:
        x = compass_net[CompassDir.E]
    else:
        x = (-1)*compass_net[CompassDir.W]

    granular_direction = ((2 * math.pi) + math.atan2(y, x)) % (2 * math.pi)
    direction_selector = round(granular_direction/options.directions_incr, 0)
    direction = direction_selector * options.directions_incr
    return direction
