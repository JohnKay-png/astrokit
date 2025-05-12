import pytest
import numpy as np
from astropy import units as u
from astrokit.core.maneuver import Maneuver
from astrokit.bodies import Earth
from astrokit.twobody.orbit import Orbit

def test_hohmann_transfer():
    """Test Hohmann transfer calculation."""
    alt_i = 400 * u.km
    alt_f = 1000 * u.km
    orbit_i = Orbit.circular(Earth, alt_i)
    orbit_f = Orbit.circular(Earth, alt_f)
    
    man = Maneuver.hohmann(orbit_i, orbit_f)
    
    # Verify delta-v values
    assert len(man.impulses) == 2
    assert np.linalg.norm(man.impulses[0][1].to(u.m/u.s)) > 100 * u.m/u.s
    assert np.linalg.norm(man.impulses[1][1].to(u.m/u.s)) > 50 * u.m/u.s

def test_bielliptic_transfer():
    """Test bielliptic transfer calculation."""
    orbit_i = Orbit.circular(Earth, 200 * u.km)
    man = Maneuver.bielliptic(orbit_i, 
                            r_b=100000 * u.km, 
                            r_f=50000 * u.km)
    
    assert len(man.impulses) == 3
    assert man.get_total_cost().to(u.km/u.s) > 4 * u.km/u.s

def test_continuous_thrust():
    """Test continuous thrust maneuver."""
    orbit = Orbit.circular(Earth, 400 * u.km)
    delta_v = 0.5 * u.km/u.s
    duration = 10 * u.min
    
    man = Maneuver.continuous_tangential(orbit, delta_v, duration)
    
    # Verify we got multiple impulses
    assert len(man.impulses) > 1
    
    # Verify total delta-v matches input
    total_dv = man.get_total_cost()
    assert u.isclose(total_dv, delta_v, rtol=0.01)
    
    # Verify duration is covered
    assert man.impulses[-1][0] <= duration
