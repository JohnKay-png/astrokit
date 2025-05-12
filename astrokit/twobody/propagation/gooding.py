from astropy import units as u

from astrokit.core.propagation.gooding import gooding_coe as gooding_fast
from astrokit.twobody.propagation.enums import PropagatorKind
from astrokit.twobody.states import ClassicalState


class GoodingPropagator:
    """Propagate the orbit using the Gooding method.

    The Gooding method solves the Elliptic Kepler Equation with a cubic convergence,
    and accuracy better than 10e-12 rad is normally achieved. It is not valid for
    eccentricities equal or greater than 1.0.

    Notes
    -----
    This method was developed by Gooding and Odell in their paper *The
    hyperbolic Kepler equation (and the elliptic equation revisited)* with
    DOI: https://doi.org/10.1007/BF01235540

    """

    kind = PropagatorKind.ELLIPTIC

    def propagate(self, state, tof):
        state = state.to_classical()

        nu = (
            gooding_fast(
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
