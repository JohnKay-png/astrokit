from astropy import units as u
import numpy as np
from numba import njit, prange
from math import sin,cos,sqrt
# Utility functions from the original code
@njit
def norm(v):
    """Compute the Euclidean norm of a vector."""
    return np.sqrt(v @ v)

@njit
def rotation_matrix(angle, axis):
    """Create a rotation matrix for a given angle about the specified axis (0=x, 1=y, 2=z)."""
    c, s = np.float64(np.cos(angle)), np.float64(np.sin(angle))
    if axis == 0:
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c]
        ], dtype=np.float64)
    elif axis == 1:
        return np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c]
        ], dtype=np.float64)
    elif axis == 2:
        return np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    else:
        raise ValueError("Axis must be 0, 1, or 2")

@njit
def eccentricity_vector(k, r, v):
    """Compute eccentricity vector."""
    return ((v @ v - k / norm(r)) * r - (r @ v) * v) / k

@njit
def circular_velocity(k, a):
    """Compute circular velocity."""
    return np.sqrt(k / a)

@njit
def rv_pqw(k, p, ecc, nu):
    """Compute position and velocity vectors in perifocal frame."""
    # Explicitly create float64 arrays
    pqw = np.array([
        [cos(nu), sin(nu), 0],
        [-sin(nu), ecc + cos(nu), 0]
    ], dtype=np.float64) * np.array([[p / (1 + ecc * cos(nu))], [sqrt(k / p)]], dtype=np.float64)
    return pqw

@njit
def coe_rotation_matrix(inc, raan, argp):
    """Create rotation matrix for classical orbital elements transformation."""
    # Explicitly create float64 rotation matrices
    r = rotation_matrix(raan, 2).astype(np.float64)
    r = (r @ rotation_matrix(inc, 0).astype(np.float64)).astype(np.float64)
    r = (r @ rotation_matrix(argp, 2).astype(np.float64)).astype(np.float64)
    return r

@njit
def coe2rv(k, p, ecc, inc, raan, argp, nu):
    """Convert classical orbital elements to state vectors."""
    # Ensure all inputs are float64
    k = np.float64(k)
    p = np.float64(p)
    ecc = np.float64(ecc)
    inc = np.float64(inc)
    raan = np.float64(raan)
    argp = np.float64(argp)
    nu = np.float64(nu)
    
    # Compute components with explicit typing
    pqw = rv_pqw(k, p, ecc, nu)
    rm = coe_rotation_matrix(inc, raan, argp)
    
    # Perform matrix multiplication with intermediate casting
    rm_t = rm.T.astype(np.float64)
    result = np.zeros((2, 3), dtype=np.float64)
    
    # Manual matrix multiplication to avoid Numba issues
    for i in range(2):
        for j in range(3):
            for k in range(3):
                result[i,j] += pqw[i,k] * rm_t[k,j]
    
    return result

@njit(parallel=False)  # Disable parallel due to potential Numba issues on some platforms
def coe2rv_many(k, p, ecc, inc, raan, argp, nu):
    """Parallel version of coe2rv."""
    n = nu.shape[0]
    rr = np.zeros((n, 3))
    vv = np.zeros((n, 3))
    for i in prange(n):
        rr[i, :], vv[i, :] = coe2rv(k[i], p[i], ecc[i], inc[i], raan[i], argp[i], nu[i])
    return rr, vv

@njit
def E_to_nu(E, ecc):
    """Convert eccentric anomaly to true anomaly."""
    return 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(E / 2), np.sqrt(1 - ecc) * np.cos(E / 2))

@njit
def F_to_nu(F, ecc):
    """Convert hyperbolic anomaly to true anomaly."""
    return 2 * np.arctan2(np.sqrt(ecc + 1) * np.sinh(F / 2), np.sqrt(ecc - 1) * np.cosh(F / 2))

@njit
def rv2coe(k, r, v, tol=1e-8):
    """Convert state vectors to classical orbital elements."""
    h = np.cross(r, v)
    n = np.cross([0, 0, 1], h)
    e = ((v @ v - k / norm(r)) * r - (r @ v) * v) / k
    ecc = norm(e)
    p = (h @ h) / k
    inc = np.arccos(h[2] / norm(h))

    circular = ecc < tol
    equatorial = abs(inc) < tol

    if equatorial and not circular:
        raan = 0
        argp = np.arctan2(e[1], e[0]) % (2 * np.pi)
        nu = np.arctan2((h @ np.cross(e, r)) / norm(h), r @ e)
    elif not equatorial and circular:
        raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
        argp = 0
        nu = np.arctan2((r @ np.cross(h, n)) / norm(h), r @ n)
    elif equatorial and circular:
        raan = 0
        argp = 0
        nu = np.arctan2(r[1], r[0]) % (2 * np.pi)
    else:
        a = p / (1 - ecc**2)
        ka = k * a
        if a > 0:
            e_se = (r @ v) / np.sqrt(ka)
            e_ce = norm(r) * (v @ v) / k - 1
            nu = E_to_nu(np.arctan2(e_se, e_ce), ecc)
        else:
            e_sh = (r @ v) / np.sqrt(-ka)
            e_ch = norm(r) * (norm(v) ** 2) / k - 1
            nu = F_to_nu(np.log((e_ch + e_sh) / (e_ch - e_sh)) / 2, ecc)

        raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
        px = r @ n
        py = (r @ np.cross(h, n)) / norm(h)
        argp = (np.arctan2(py, px) - nu) % (2 * np.pi)

    nu = (nu + np.pi) % (2 * np.pi) - np.pi
    return p, ecc, inc, raan, argp, nu

# New functions with Astropy units
def mean_motion(k, a):
    """Compute mean motion from gravitational parameter and semimajor axis."""
    k = k.to(u.km**3 / u.s**2)
    a = a.to(u.km)
    return np.sqrt(k / a**3).to(u.rad / u.s)

def period(k, a):
    """Compute orbital period from gravitational parameter and semimajor axis."""
    k = k.to(u.km**3 / u.s**2)
    a = a.to(u.km)
    return (2 * np.pi * np.sqrt(a**3 / k)).to(u.s)

def t_p(k, a, ecc, nu):
    """Compute time since periapsis from true anomaly."""
    k = k.to(u.km**3 / u.s**2)
    a = a.to(u.km)
    ecc = ecc.to(u.dimensionless_unscaled)
    nu = nu.to(u.rad)
    E = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(nu / 2))
    M = E - ecc * np.sin(E)
    return (M / mean_motion(k, a)).to(u.s)

class ClassicalElements:
    def __init__(self, body, a, ecc, inc, raan, argp, nu):
        """Classical orbital elements."""
        self.body = body
        self.a = a.to(u.km)
        self.ecc = ecc.to(u.dimensionless_unscaled)
        self.inc = inc.to(u.rad)
        self.raan = raan.to(u.rad)
        self.argp = argp.to(u.rad)
        self.nu = nu.to(u.rad)

    @property
    def p(self):
        """Semi-latus rectum."""
        return self.a * (1 - self.ecc**2)

    @property
    def r_p(self):
        """Radius of periapsis."""
        return self.a * (1 - self.ecc)

    @property
    def r_a(self):
        """Radius of apoapsis."""
        return self.a * (1 + self.ecc)

    def to_vectors(self, k=None):
        """Convert classical orbital elements to position and velocity vectors."""
        k = self.body.k if k is None else k
        # Explicitly convert to float64 before Numba
        k = np.float64(k.to(u.km**3 / u.s**2).value)
        p = np.float64(self.p.to(u.km).value)
        ecc = np.float64(self.ecc.value)
        inc = np.float64(self.inc.to(u.rad).value)
        raan = np.float64(self.raan.to(u.rad).value)
        argp = np.float64(self.argp.to(u.rad).value)
        nu = np.float64(self.nu.to(u.rad).value)

        # Compute vectors in inertial frame using coe2rv
        try:
            ijk = coe2rv(k, p, ecc, inc, raan, argp, nu)
            r = ijk[0, :] * u.km
            v = ijk[1, :] * u.km / u.s
            return r, v
        except Exception as e:
            raise ValueError(
                f"Failed to convert elements to vectors: {e}\n"
                f"Types: k={type(k)}, p={type(p)}, ecc={type(ecc)}\n"
                f"Values: k={k}, p={p}, ecc={ecc}"
            ) from e

    @classmethod
    def from_vectors(cls, body, r, v, k=None):
        """Create classical orbital elements from position and velocity vectors."""
        k = body.k if k is None else k
        k = k.to(u.km**3 / u.s**2).value
        r = r.to(u.km).value
        v = v.to(u.km / u.s).value

        # Compute classical elements using rv2coe
        p, ecc, inc, raan, argp, nu = rv2coe(k, r, v)
        a = p / (1 - ecc**2)  # Compute semimajor axis
        return cls(
            body,
            a=a * u.km,
            ecc=ecc * u.dimensionless_unscaled,
            inc=inc * u.rad,
            raan=raan * u.rad,
            argp=argp * u.rad,
            nu=nu * u.rad
        )

__all__ = ["ClassicalElements", "mean_motion", "period", "t_p", "eccentricity_vector", 
           "circular_velocity", "rv_pqw", "coe2rv", "coe2rv_many", "rv2coe"]
