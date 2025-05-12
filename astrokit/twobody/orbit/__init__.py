import numpy as np
from astropy import units as u
from astrokit.twobody.elements import ClassicalElements
from ..propagation import propagate
from ..maneuver import impulse_maneuver
from ...bodies import Body

class Orbit:
    def __init__(self, body, elements):
        self.body = body
        self.elements = elements
        
    @classmethod
    def circular(cls, body, alt):
        """Create circular orbit around body at given altitude"""
        a = body.R + alt
        elements = ClassicalElements(
            a=a,
            ecc=0 * u.one,
            inc=0 * u.deg,
            raan=0 * u.deg,
            argp=0 * u.deg,
            nu=0 * u.deg
        )
        return cls(body, elements)
        
    @classmethod 
    def from_body_ephem(cls, body):
        """Create orbit from body's ephemeris"""
        # TODO: Implement ephemeris lookup
        return cls.circular(body, 0 * u.km)
        
    def propagate(self, time, propagator=None):
        """Propagate orbit for given time duration
        
        Parameters:
        time : Quantity
            Time duration to propagate
        propagator : object, optional
            Propagator to use for the propagation
        """
        new_elements = propagate(self.elements, time, propagator=propagator)
        return Orbit(self.body, new_elements)
        
    def impulse_maneuver(self, target_orbit):
        """Calculate impulse maneuver to reach target orbit"""
        return impulse_maneuver(self.elements, target_orbit.elements)
        
    @property
    def a(self):
        return self.elements.a
        
    @property
    def ecc(self):
        return self.elements.ecc
        
    @property
    def period(self):
        return 2 * np.pi * np.sqrt(self.a**3 / self.body.k)
        
    @property
    def rv(self):
        """Get position and velocity vectors"""
        return self.elements.to_vectors(self.body.k.to(u.km**3 / u.s**2))

__all__ = ["Orbit"]
