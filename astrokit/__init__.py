"""Utilities and Python wrappers for Orbital Mechanics."""

__version__ = "1.0.0"

# 延迟导入以避免循环依赖
def __getattr__(name):
    if name in {'Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 
               'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Moon'}:
        from astrokit.bodies import __dict__ as bodies_dict
        return bodies_dict[name]
    raise AttributeError(f"module 'astrokit' has no attribute '{name}'")

__all__ = [
    'Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 
    'Uranus', 'Neptune', 'Pluto', 'Moon', 'Orbit', 'Maneuver', 'OrbitPlotter'
]
