"""Bodies of the Solar System.

Contains some predefined bodies of the Solar System:

* Sun (☉)
* Earth (♁)
* Moon (☾)
* Mercury (☿)
* Venus (♀)
* Mars (♂)
* Jupiter (♃)
* Saturn (♄)
* Uranus (⛢)
* Neptune (♆)
* Pluto (♇)
* Phobos
* Deimos
* Europa
* Ganyemede
* Enceladus
* Titan
* Titania
* Triton
* Charon


and a way to define new bodies (:py:class:`~Body` class).

Data references can be found in astrokit.constants
"""
from dataclasses import dataclass
import math

from astropy import units as u
from astropy.constants import G
from astropy.units import Quantity

from astrokit.constants import (
    GM_sun, R_sun, J2_sun, M_sun,
    GM_mercury, R_mercury, R_mean_mercury, R_polar_mercury,
    GM_venus, R_venus, R_mean_venus, R_polar_venus, J2_venus, J3_venus,
    GM_earth, R_earth, R_mean_earth, R_polar_earth, J2_earth, J3_earth, M_earth,
    GM_mars, R_mars, R_mean_mars, R_polar_mars, J2_mars, J3_mars,
    GM_jupiter, R_jupiter, R_mean_jupiter, R_polar_jupiter, M_jupiter,
    GM_saturn, R_saturn, R_mean_saturn, R_polar_saturn,
    GM_uranus, R_uranus, R_mean_uranus, R_polar_uranus,
    GM_neptune, R_neptune, R_mean_neptune, R_polar_neptune,
    GM_pluto, R_pluto, R_mean_pluto, R_polar_pluto,
    GM_moon, R_moon, R_mean_moon, R_polar_moon,
    GM_phobos, GM_deimos, GM_europa, GM_ganymede, GM_enceladus,
    GM_titan, GM_titania, GM_triton, GM_charon,
    rotational_period_sun, rotational_period_mercury, rotational_period_venus,
    rotational_period_earth, rotational_period_moon, rotational_period_mars,
    rotational_period_jupiter, rotational_period_saturn, rotational_period_uranus,
    rotational_period_neptune, rotational_period_pluto,
    mean_a_mercury, mean_a_venus, mean_a_earth, mean_a_mars, mean_a_jupiter,
    mean_a_saturn, mean_a_uranus, mean_a_neptune, mean_a_moon, mean_a_phobos,
    mean_a_deimos, mean_a_europa, mean_a_ganymede, mean_a_enceladus,
    mean_a_titan, mean_a_titania, mean_a_triton, mean_a_charon
)
from astrokit.frames import Planes


@dataclass
class Body:
    """Class representing a celestial body."""
    parent: 'Body'
    k: Quantity
    name: str
    symbol: str = None
    R: Quantity = 0 * u.km
    R_polar: Quantity = 0 * u.km
    R_mean: Quantity = 0 * u.km
    rotational_period: Quantity = 0.0 * u.day
    J2: Quantity = 0.0 * u.one
    J3: Quantity = 0.0 * u.one
    mass: Quantity = None
    mean_a: Quantity = 0.0 * u.km

    def __post_init__(self):
        if self.mass is None:
            self.mass = self.k / G

    @property
    def angular_velocity(self):
        return (2 * math.pi * u.rad) / self.rotational_period.to(u.s)

    def __str__(self):
        return f"{self.name} ({self.symbol})"

    def __reduce__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @classmethod
    @u.quantity_input(k=u.km**3 / u.s**2, R=u.km)
    def from_parameters(cls, parent, k, name, symbol, R, **kwargs):
        return cls(parent, k, name, symbol, R, **kwargs)

    @classmethod
    def from_relative(
        cls, reference, parent, k, name, symbol=None, R=0, **kwargs
    ):
        k = k * reference.k
        R = R * reference.R
        return cls(parent, k, name, symbol, R, **kwargs)


Sun = Body(
    parent=None,
    k=GM_sun,
    name="Sun",
    symbol="\u2609",
    R=R_sun,
    rotational_period=rotational_period_sun,
    J2=J2_sun,
    mass=M_sun,
)


Mercury = Body(
    parent=Sun,
    k=GM_mercury,
    name="Mercury",
    symbol="\u263F",
    R=R_mercury,
    R_mean=R_mean_mercury,
    R_polar=R_polar_mercury,
    rotational_period=rotational_period_mercury,
    mean_a=mean_a_mercury,
)


Venus = Body(
    parent=Sun,
    k=GM_venus,
    name="Venus",
    symbol="\u2640",
    R=R_venus,
    R_mean=R_mean_venus,
    R_polar=R_polar_venus,
    rotational_period=rotational_period_venus,
    J2=J2_venus,
    J3=J3_venus,
    mean_a=mean_a_venus,
)


Earth = Body(
    parent=Sun,
    k=GM_earth,
    name="Earth",
    symbol="\u2641",
    R=R_earth,
    R_mean=R_mean_earth,
    R_polar=R_polar_earth,
    rotational_period=rotational_period_earth,
    mass=M_earth,
    J2=J2_earth,
    J3=J3_earth,
    mean_a=mean_a_earth,
)


Mars = Body(
    parent=Sun,
    k=GM_mars,
    name="Mars",
    symbol="\u2642",
    R=R_mars,
    R_mean=R_mean_mars,
    R_polar=R_polar_mars,
    rotational_period=rotational_period_mars,
    J2=J2_mars,
    J3=J3_mars,
    mean_a=mean_a_mars,
)


Jupiter = Body(
    parent=Sun,
    k=GM_jupiter,
    name="Jupiter",
    symbol="\u2643",
    R=R_jupiter,
    R_mean=R_mean_jupiter,
    R_polar=R_polar_jupiter,
    rotational_period=rotational_period_jupiter,
    mass=M_jupiter,
    mean_a=mean_a_jupiter,
)


Saturn = Body(
    parent=Sun,
    k=GM_saturn,
    name="Saturn",
    symbol="\u2644",
    R=R_saturn,
    R_mean=R_mean_saturn,
    R_polar=R_polar_saturn,
    rotational_period=rotational_period_saturn,
    mean_a=mean_a_saturn,
)


Uranus = Body(
    parent=Sun,
    k=GM_uranus,
    name="Uranus",
    symbol="\u26E2",
    R=R_uranus,
    R_mean=R_mean_uranus,
    R_polar=R_polar_uranus,
    rotational_period=rotational_period_uranus,
    mean_a=mean_a_uranus,
)


Neptune = Body(
    parent=Sun,
    k=GM_neptune,
    name="Neptune",
    symbol="\u2646",
    R=R_neptune,
    R_mean=R_mean_neptune,
    R_polar=R_polar_neptune,
    rotational_period=rotational_period_neptune,
    mean_a=mean_a_neptune,
)


Pluto = Body(
    parent=Sun,
    k=GM_pluto,
    name="Pluto",
    symbol="\u2647",
    R=R_pluto,
    R_mean=R_mean_pluto,
    R_polar=R_polar_pluto,
    rotational_period=rotational_period_pluto,
)  # No mean_a_pluto as Pluto is officially not a planet around Sun


Moon = Body(
    parent=Earth,
    k=GM_moon,
    name="Moon",
    symbol="\u263E",
    R=R_moon,
    R_mean=R_mean_moon,
    R_polar=R_polar_moon,
    rotational_period=rotational_period_moon,
    mean_a=mean_a_moon,
)


Phobos = Body(
    parent=Mars,
    k=GM_phobos,
    name="Phobos",
    mean_a=mean_a_phobos,
)

Deimos = Body(
    parent=Mars,
    k=GM_deimos,
    name="Deimos",
    mean_a=mean_a_deimos,
)

Europa = Body(
    parent=Jupiter,
    k=GM_europa,
    name="Europa",
    mean_a=mean_a_europa,
)

Ganymede = Body(
    parent=Jupiter,
    k=GM_ganymede,
    name="Ganymede",
    mean_a=mean_a_ganymede,
)

Enceladus = Body(
    parent=Saturn,
    k=GM_enceladus,
    name="Enceladus",
    mean_a=mean_a_enceladus,
)

Titan = Body(
    parent=Saturn,
    k=GM_titan,
    name="Titan",
    mean_a=mean_a_titan,
)

Titania = Body(
    parent=Uranus,
    k=GM_titania,
    name="Titania",
    mean_a=mean_a_titania,
)

Triton = Body(
    parent=Neptune,
    k=GM_triton,
    name="Triton",
    mean_a=mean_a_triton,
)

Charon = Body(
    parent=Pluto,
    k=GM_charon,
    name="charon",
    mean_a=mean_a_charon,
)
