import sys
import numpy as np
from astropy import units as u
from ..elements import ClassicalElements
from ..states import ClassicalState
from ._compat import OldPropagatorModule

sys.modules[__name__] = OldPropagatorModule(sys.modules[__name__])

class PimientaPropagator:
    """Basic Kepler solver for elliptic orbits."""

    def propagate(self, state, tof):
        """Basic implementation of Pimienta's algorithm.
        
        Args:
            state: Classical orbital state
            tof: Time of flight (astropy Quantity)
            
        Returns:
            Propagated classical orbital state
        """
        elements = state.to_classical()
        
        # Simplified Pimienta algorithm implementation
        mean_motion = np.sqrt(state.attractor.k / elements.a**3)
        delta_nu = (mean_motion * tof).to(u.rad)
        
        new_nu = elements.nu + delta_nu
        
        return ClassicalState(
            state.attractor,
            (elements.a, elements.ecc, elements.inc, 
             elements.raan, elements.argp, new_nu),
            state.plane
        )
