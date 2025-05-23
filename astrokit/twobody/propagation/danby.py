from astropy import units as u

from astrokit.core.propagation import danby as danby_fast
from astrokit.twobody.propagation.enums import PropagatorKind
from astrokit.twobody.states import ClassicalState


class DanbyPropagator:
    """Kepler solver for both elliptic and parabolic orbits based on Danby's algorithm.

    Notes
    -----
    This algorithm was developed by Danby in his paper *The solution of Kepler
    Equation* with DOI: https://doi.org/10.1007/BF01686811

    """

    kind = PropagatorKind.ELLIPTIC | PropagatorKind.HYPERBOLIC

    def propagate(self, state, tof):
        state = state.to_classical()

        nu = (
            danby_fast(
                state.attractor.k.to_value(u.km**3 / u.s**2),
                *state.to_value(),
                tof.to_value(u.s),
            )
            << u.rad
        )

        new_state = ClassicalState(
            state.attractor, state.to_tuple()[:5] + (nu,), state.plane
        )
        return new_state
