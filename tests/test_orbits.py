import pytest
import numpy as np
from astropy import units as u
from astrokit.twobody.orbit import Orbit
from astrokit.bodies import Earth, Mars

def test_circular_orbit():
    # Test basic circular orbit parameters
    orbit = Orbit.circular(Earth, alt=400 * u.km)
    assert orbit.ecc == 0 * u.one
    assert orbit.a == Earth.R + 400 * u.km
    assert orbit.period.to(u.minute) > 90 * u.minute

def test_orbit_maneuver():
    # Test orbit maneuver calculation
    initial = Orbit.circular(Earth, alt=400 * u.km)
    final = Orbit.circular(Earth, alt=1000 * u.km)
    dv = initial.impulse_maneuver(final)
    assert dv[0].to(u.m/u.s) > 100 * u.m/u.s

def test_mars_orbit_propagation():
    # Test Mars orbit propagation
    orbit = Orbit.from_body_ephem(Mars)
    new_orbit = orbit.propagate(30 * u.day)
    # Simple check that propagation worked
    assert new_orbit.elements.a == orbit.elements.a
