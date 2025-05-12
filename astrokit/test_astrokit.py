"""Test script for astrokit library."""
from astropy import time, units as u
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from bodies import Earth, Sun
from twobody import Orbit

def test_iss_orbit():
    """Test ISS orbit calculation."""
    iss = Orbit.from_vectors(
        Earth,
        [8.59072560e2, -4.13720368e3, 5.29556871e3] * u.km,
        [7.37289205, 2.08223573, 4.39999794e-1] * u.km / u.s,
        time.Time("2013-03-18 12:00", scale="utc"),
    )
    print(f"ISS Orbit: {iss}")
    return iss

def test_molniya_orbit():
    """Test Molniya orbit calculation."""
    molniya = Orbit.from_classical(
        attractor=Earth,
        a=26600 * u.km,
        ecc=0.75 * u.one,
        inc=63.4 * u.deg,
        raan=0 * u.deg,
        argp=270 * u.deg,
        nu=80 * u.deg,
    )
    print(f"Molniya Orbit: {molniya}")
    return molniya

def plot_orbit(orbit):
    """Plot an orbit."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    orbit.plot_3d(ax=ax)
    plt.title(f"{orbit.attractor.name} Orbit")
    plt.show()

if __name__ == "__main__":
    print("Testing astrokit library functionality")
    iss = test_iss_orbit()
    molniya = test_molniya_orbit()
    plot_orbit(iss)
    plot_orbit(molniya)
