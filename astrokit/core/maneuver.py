"""Maneuver implementations including the Maneuver class."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from numba import njit as jit
from astropy import units as u
import numpy as np
from numpy import cross

from numpy.linalg import norm
from astrokit.core.elements import coe_rotation_matrix, rv2coe, rv_pqw
from astrokit.twobody.orbit import Orbit

@dataclass
class Impulse:
    """Single impulse in a maneuver."""
    delta_v: u.Quantity[u.km/u.s]
    time: u.Quantity[u.s]

@dataclass
class Maneuver:
    """Collection of impulses making up a maneuver."""
    impulses: List[Tuple[u.Quantity[u.s], u.Quantity[u.km/u.s]]]
    
    @classmethod
    def hohmann(cls, orbit_i: Orbit, orbit_f: Orbit) -> 'Maneuver':
        """Create Hohmann transfer maneuver between two orbits.
        
        Parameters
        ----------
        orbit_i : Orbit
            Initial orbit
        orbit_f : Orbit
            Final orbit
            
        Returns
        -------
        Maneuver
            Hohmann transfer maneuver with two impulses
        """
        # Extract numerical values in consistent units
        k = orbit_i.body.k.to_value(u.km**3/u.s**2)
        
        # Get position in km and velocity in km/s
        r = orbit_i.rv[0].to_value(u.km)  # position vector
        v = orbit_i.rv[1].to_value(u.km/u.s)  # velocity vector
        r_f = orbit_f.a.to_value(u.km)
        
        dv_a, dv_b, t_trans = hohmann(k, r, v, r_f)
        
        # Return with proper units
        return cls([
            (0.0 * u.s, dv_a * u.km/u.s),
            (t_trans * u.s, dv_b * u.km/u.s)
        ])
        
    @classmethod
    def bielliptic(cls, orbit_i: Orbit, r_b: u.Quantity[u.km], 
                 r_f: u.Quantity[u.km]) -> 'Maneuver':
        """Create bielliptic transfer maneuver.
        
        Parameters
        ----------
        orbit_i : Orbit
            Initial orbit
        r_b : Quantity
            Radius of intermediate orbit (km)
        r_f : Quantity
            Radius of final orbit (km)
            
        Returns
        -------
        Maneuver
            Bielliptic transfer maneuver with three impulses
        """
        # Extract numerical values in consistent units
        k = orbit_i.body.k.to_value(u.km**3/u.s**2)
        
        # Get position in km and velocity in km/s
        r = orbit_i.rv[0].to_value(u.km)  # position vector
        v = orbit_i.rv[1].to_value(u.km/u.s)  # velocity vector
        r_b_val = r_b.to_value(u.km)
        r_f_val = r_f.to_value(u.km)
        
        dv_a, dv_b, dv_c, t1, t2 = bielliptic(k, r_b_val, r_f_val, (r, v))
        
        # Return with proper units
        return cls([
            (0.0 * u.s, dv_a * u.km/u.s),
            (t1 * u.s, dv_b * u.km/u.s),
            ((t1 + t2) * u.s, dv_c * u.km/u.s)
        ])
        
    def get_total_cost(self) -> u.Quantity[u.km/u.s]:
        """Calculate total delta-v for the maneuver.
        
        Returns
        -------
        Quantity
            Total delta-v cost of maneuver (km/s)
        """
        total = 0.0 * u.km/u.s
        for _, dv in self.impulses:
            total += norm(dv)
        return total

    @classmethod
    def continuous_tangential(cls, orbit: Orbit, 
                            delta_v: u.Quantity[u.km/u.s],
                            duration: u.Quantity[u.s]) -> 'Maneuver':
        """Create continuous tangential thrust maneuver.
        
        Parameters
        ----------
        orbit : Orbit
            Initial orbit
        delta_v : Quantity
            Total delta-v to apply (km/s)
        duration : Quantity
            Duration over which to apply thrust (s)
            
        Returns
        -------
        Maneuver
            Continuous thrust maneuver with multiple small impulses
        """
        # Ensure proper units
        delta_v = delta_v.to(u.km/u.s)
        duration = duration.to(u.s)
        
        # Calculate thrust acceleration (km/s^2)
        a_t = (delta_v / duration).to(u.km/u.s**2)
        
        # Split into small impulses (minimum 10, or more for longer durations)
        n_impulses = max(10, int(float(duration / orbit.period) * 10))
        impulses = []
        
        for i in range(n_impulses):
            t = duration * i / n_impulses
            dv = a_t * (duration / n_impulses)
            impulses.append((t, dv))
            
        return cls(impulses)


@jit
def hohmann(k, r, v, r_f):
    r"""Calculate the Hohmann maneuver velocities and the duration of the maneuver.

    By defining the relationship between orbit radius:

    .. math::
        a_{trans} = \frac{r_{i} + r_{f}}{2}

    The Hohmann maneuver velocities can be expressed as:

    .. math::
        \begin{align}
            \Delta v_{a} &= \sqrt{\frac{2\mu}{r_{i}} - \frac{\mu}{a_{trans}}} - v_{i}\\
            \Delta v_{b} &= \sqrt{\frac{\mu}{r_{f}}} - \sqrt{\frac{2\mu}{r_{f}} - \frac{\mu}{a_{trans}}}
        \end{align}

    The time that takes to complete the maneuver can be computed as:

    .. math::
        \tau_{trans} = \pi \sqrt{\frac{(a_{trans})^{3}}{\mu}}

    Parameters
    ----------
    k : float
        Standard Gravitational parameter
    r : numpy.ndarray
        Position vector (km)
    v : numpy.ndarray
        Velocity vector (km/s)
    r_f : float
        Final orbital radius (km)

    """
    rv = (r, v)
    _, ecc, inc, raan, argp, nu = rv2coe(k, *rv)
    h_i = norm(cross(*rv))
    p_i = h_i**2 / k

    r_i, v_i = rv_pqw(k, p_i, ecc, nu)

    r_i = norm(r_i)
    v_i = norm(v_i)
    a_trans = (r_i + r_f) / 2

    dv_a = np.sqrt(2 * k / r_i - k / a_trans) - v_i
    dv_b = np.sqrt(k / r_f) - np.sqrt(2 * k / r_f - k / a_trans)

    dv_a = np.array([0, dv_a, 0])
    dv_b = np.array([0, -dv_b, 0])

    rot_matrix = coe_rotation_matrix(inc, raan, argp)

    dv_a = rot_matrix @ dv_a
    dv_b = rot_matrix @ dv_b

    t_trans = np.pi * np.sqrt(a_trans**3 / k)

    return dv_a, dv_b, t_trans


@jit
def bielliptic(k, r_b, r_f, rv):
    r"""Calculate the increments in the velocities and the time of flight of the maneuver.

    The bielliptic maneuver employs two Hohmann transfers, therefore two
    intermediate orbits are established. We define the different radius
    relationships as follows:

    .. math::
        \begin{align}
            a_{trans1} &= \frac{r_{i} + r_{b}}{2}\\
            a_{trans2} &= \frac{r_{b} + r_{f}}{2}\\
        \end{align}

    The increments in the velocity are:

    .. math::
        \begin{align}
            \Delta v_{a} &= \sqrt{\frac{2\mu}{r_{i}} - \frac{\mu}{a_{trans1}}} - v_{i}\\
            \Delta v_{b} &= \sqrt{\frac{2\mu}{r_{b}} - \frac{\mu}{a_{trans2}}} - \sqrt{\frac{2\mu}{r_{b}} - \frac{\mu}{a_trans{1}}}\\
            \Delta v_{c} &= \sqrt{\frac{\mu}{r_{f}}} - \sqrt{\frac{2\mu}{r_{f}} - \frac{\mu}{a_{trans2}}}\\
        \end{align}

    The time of flight for this maneuver is the addition of the time needed for both transition orbits, following the same formula as
    Hohmann:

    .. math::
        \begin{align}
            \tau_{trans1} &= \pi \sqrt{\frac{a_{trans1}^{3}}{\mu}}\\
            \tau_{trans2} &= \pi \sqrt{\frac{a_{trans2}^{3}}{\mu}}\\
        \end{align}

    Parameters
    ----------
    k : float
        Standard Gravitational parameter
    r_b : float
        Altitude of the intermediate orbit
    r_f : float
        Final orbital radius
    rv : numpy.ndarray, numpy.ndarray
        Position and velocity vectors

    """
    _, ecc, inc, raan, argp, nu = rv2coe(k, *rv)
    h_i = norm(cross(*rv))
    p_i = h_i**2 / k

    r_i, v_i = rv_pqw(k, p_i, ecc, nu)

    r_i = norm(r_i)
    v_i = norm(v_i)
    a_trans1 = (r_i + r_b) / 2
    a_trans2 = (r_b + r_f) / 2

    dv_a = np.sqrt(2 * k / r_i - k / a_trans1) - v_i
    dv_b = np.sqrt(2 * k / r_b - k / a_trans2) - np.sqrt(
        2 * k / r_b - k / a_trans1
    )
    dv_c = np.sqrt(k / r_f) - np.sqrt(2 * k / r_f - k / a_trans2)

    dv_a = np.array([0, dv_a, 0])
    dv_b = np.array([0, -dv_b, 0])
    dv_c = np.array([0, dv_c, 0])

    rot_matrix = coe_rotation_matrix(inc, raan, argp)

    dv_a = rot_matrix @ dv_a
    dv_b = rot_matrix @ dv_b
    dv_c = rot_matrix @ dv_c

    t_trans1 = np.pi * np.sqrt(a_trans1**3 / k)
    t_trans2 = np.pi * np.sqrt(a_trans2**3 / k)

    return dv_a, dv_b, dv_c, t_trans1, t_trans2


@jit
def correct_pericenter(k, R, J2, max_delta_r, v, a, inc, ecc):
    """Calculates the time before burning and the velocity vector in direction of the burn.

    Parameters
    ----------
    k : float
        Standard Gravitational parameter
    R : float
        Radius of the attractor
    J2 : float
        Oblateness factor
    max_delta_r : float
        Maximum satellite’s geocentric distance
    v : numpy.ndarray
        Velocity vector
    a : float
        Semi-major axis
    inc : float
        Inclination
    ecc : float
        Eccentricity

    Notes
    -----
    The algorithm was obtained from "Fundamentals of Astrodynamics and Applications, 4th ed (2013)" by David A.
    Vallado, page 885.
    Given a max_delta_r, we determine the maximum perigee drift before we do an orbit-adjustment burn
    to restore the perigee to its nominal value. We estimate the time until this burn using the allowable drift
    delta_w and the drift rate :math:`|dw|`.
    For positive delta_v, the change in the eccentricity is positive for perigee burns and negative for apogee burns.
    The opposite holds for a delta_v applied against the velocity vector, which decreases the satellite’s velocity.
    Perigee drift are mainly due to the zonal harmonics, which cause variations in the altitude by changing the
    argument of perigee.
    Please note that ecc ≈ 0.001, so the error incurred by assuming a small eccentricity is on the order of 0.1%.
    This is smaller than typical variations in thruster performance between burns.

    """
    p = a * (1 - ecc**2)
    n = (k / a**3) ** 0.5

    dw = ((3 * n * R**2 * J2) / (4 * p**2)) * (4 - 5 * np.sin(inc) ** 2)

    delta_w = 2 * (1 + ecc) * max_delta_r
    delta_w /= a * ecc * (1 - ecc)
    delta_w **= 0.5
    delta_t = abs(delta_w / dw)
    delta_v = 0.5 * n * a * ecc * abs(delta_w)

    vf_ = v / norm(v) * delta_v

    return delta_t, vf_
