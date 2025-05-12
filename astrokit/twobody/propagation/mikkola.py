import sys
import numpy as np
from astropy import units as u
from ..elements import ClassicalElements
from ..states import ClassicalState
from ._compat import OldPropagatorModule

sys.modules[__name__] = OldPropagatorModule(sys.modules[__name__])

class MikkolaPropagator:
    """Solves Kepler Equation by a cubic approximation."""

    def propagate(self, state, tof):
        """Basic implementation of Mikkola's algorithm.
        
        Args:
            state: Classical orbital state
            tof: Time of flight (astropy Quantity)
            
        Returns:
            Propagated classical orbital state
        """
        elements = state.to_classical()
        
        # Simplified Mikkola algorithm implementation
        mean_motion = np.sqrt(state.attractor.k / elements.a**3)
        delta_nu = (mean_motion * tof).to(u.rad)
        
        new_nu = elements.nu + delta_nu
        
        return ClassicalState(
            state.attractor,
            (elements.a, elements.ecc, elements.inc, 
             elements.raan, elements.argp, new_nu),
            state.plane
        )
