from enum import Enum

class Enums:
     pass

class NodeSet(Enum):
     OUTER = 1
     DETAIL = 2
     OUTEROBLIT = 3
     DETAILOBLIT = 4

class Direction(Enum):
     FORWARD = 1
     BACKWARD = 2

class CompassDir(Enum):
     N = 1
     E = 2
     S = 3
     W = 4

class CompassType(Enum):
     legality_compass = 1
     proximity_compass = 2
     intersects_compass = 3
     outer_attraction_compass = 4
     parallels_compass = 5
     deflection_compass = 6
     inner_attraction = 7