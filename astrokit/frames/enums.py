from enum import Enum

class Planes(Enum):
    """Reference planes for coordinate systems."""
    EARTH_EQUATOR = "EARTH_EQUATOR"
    ECLIPTIC = "ECLIPTIC"
    BODY_EQUATOR = "BODY_EQUATOR"
    ORBITAL = "ORBITAL"
