import os
import sys
import pytest
from astropy import units as u

# 将项目根目录添加到Python路径，
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from astrokit.bodies import Earth, Mars, Sun, Body
from astrokit.constants import (
    R_earth, R_polar_earth, J2_earth,
    R_mars,
    R_sun
)

def test_earth_parameters():
    """Test Earth's physical parameters"""
    assert Earth.R == R_earth
    assert Earth.R_polar == R_polar_earth
    assert Earth.J2 == J2_earth
    assert isinstance(Earth, Body)

def test_mars_parameters():
    """Test Mars' physical parameters"""
    assert Mars.R == R_mars
    assert isinstance(Mars, Body)

def test_sun_properties():
    """Test Sun's physical properties"""
    assert Sun.R == R_sun
    assert isinstance(Sun, Body)
    assert Sun.parent is None

def test_dataclass_behavior():
    """Test Body dataclass functionality"""
    test_body = Body(
        parent=Earth,
        k=1.0 * u.km**3 / u.s**2,
        name="Test Body",
        symbol="T"
    )
    assert test_body.name == "Test Body"
    assert test_body.parent == Earth


if __name__ == "__main__":
    pytest.main()
