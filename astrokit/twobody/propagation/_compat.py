from astropy import units as u
import numpy as np
from ..elements import ClassicalElements
import sys
from types import ModuleType
from .enums import PropagatorKind
from .cowell import CowellPropagator

class OldPropagatorModule:
    """兼容旧版传播器模块的包装类"""
    
    def __init__(self, module):
        self._module = module
        # Copy module attributes to wrapper
        self.__dict__.update(module.__dict__)
        
    def __getattr__(self, name):
        return getattr(self._module, name)

def propagate(elements, time, propagator=None):
    """Propagate orbital elements forward in time
    
    Parameters:
    elements : ClassicalElements
        Initial orbital elements
    time : Quantity
        Time duration to propagate
    propagator : object, optional
        Propagator instance to use. If None, uses simple dummy propagation.
        
    Returns:
    ClassicalElements: Propagated orbital elements
    """
    if propagator is not None:
        # Pass central body's gravitational parameter to propagator
        if isinstance(propagator, CowellPropagator):
            # Cowell propagator gets k from state.attractor.k internally
            return propagator.propagate(elements, time)
        else:
            # Other propagators may need k parameter
            return propagator.propagate(elements, time, k=elements.body.k)
        
    # Fallback to simple propagation if no propagator specified
    new_elements = ClassicalElements(
        a=elements.a,
        ecc=elements.ecc,
        inc=elements.inc,
        raan=elements.raan,
        argp=elements.argp,
        nu=elements.nu + 0.1 * u.deg  # Small dummy change
    )
    # Preserve body reference
    if hasattr(elements, 'body'):
        new_elements.body = elements.body
    return new_elements

__all__ = ["propagate"]
