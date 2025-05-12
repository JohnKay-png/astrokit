import pytest
from astropy import units as u
from astrokit.bodies import Earth
from astrokit.twobody import Orbit
from astrokit.twobody.elements import ClassicalElements
from astrokit.twobody.propagation import CowellPropagator
from astrokit.core.perturbations import J2_perturbation

def test_j2_perturbation_effects():
    """Test that J2 perturbations produce expected secular effects."""
    # Setup initial orbit
    r0 = [-2384.46, 5729.01, 3050.46] * u.km
    v0 = [-7.36138, -2.98997, 1.64354] * u.km / u.s
    elements = ClassicalElements.from_vectors(r0, v0, Earth.k)
    initial_orbit = Orbit(Earth, elements)

    # Propagation parameters
    tof = (48.0 * u.h).to(u.s)
    J2 = Earth.J2.value
    R = Earth.R.to(u.km).value

    # Configure Cowell propagator with J2 perturbations
    propagator = CowellPropagator(
        f=lambda t, state: J2_perturbation(t, state, Earth.k.value, J2, R))

    # Propagate orbit
    final_orbit = initial_orbit.propagate(tof)

    # Calculate rates
    raan_rate = ((final_orbit.raan - initial_orbit.raan) / tof).to(u.deg / u.day)
    argp_rate = ((final_orbit.argp - initial_orbit.argp) / tof).to(u.deg / u.day)

    # Verify expected secular effects from J2
    assert raan_rate.value < 0  # RAAN should regress
    assert abs(raan_rate.value) > 0.1  # Significant rate
    assert argp_rate.value > 0  # Argument of perigee should advance
