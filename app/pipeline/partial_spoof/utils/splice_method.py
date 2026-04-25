"""
Splice method enumeration and weighted selection proportions.

Defines the seven word-replacement splice techniques used by the partial-spoof
pipeline and their production sampling weights. Weights are calibrated to match
the acoustic diversity distribution described in investigation.md Section 8.
"""
from enum import Enum
from typing import Dict


class SpliceMethod(str, Enum):
    """Enumeration of supported splice boundary techniques.

    Each variant corresponds to a distinct fade-curve shape (or no fade) applied
    at word-replacement boundaries during partial-spoof audio generation. The
    string values are recorded verbatim in splice_metadata.json for traceability.

    Variants:
        CUT_PASTE: Direct concatenation at zero-crossings. No fade. Creates the
            sharpest boundary discontinuity; maximum forensic detectability.
        OLA_HANNING: Overlap-Add with Hann window. Equal-gain S-curve blend.
            Smoothest perceptual transition; matches signal processing literature.
        LINEAR: Equal-gain linear blend. Simple diagonal fade; creates an
            amplitude dip at the midpoint (sum = 1 but psychoacoustically lower).
        COSINE: Equal-power quarter-cosine. fade_in = sin(pi*t/2). Perceptually
            constant loudness; starts gradually, accelerates toward end.
        HALF_SINE: Equal-power square-root law. fade_in = sqrt(t). Rises faster
            than cosine initially then plateaus; common in broadcast production.
        LOGARITHMIC: Logarithmic fade_in = log(1+9t)/log(10). Aggressive initial
            rise then plateau; creates a distinctive energy artifact.
        PARABOLA: Inverted parabola fade_in = 1-(1-t)^2. Equal-gain concave
            blend; intermediate between linear and Hann in perceptual smoothness.
    """

    CUT_PASTE = "cut_paste"
    OLA_HANNING = "ola_hanning"
    LINEAR = "linear"
    COSINE = "cosine"
    HALF_SINE = "half_sine"
    LOGARITHMIC = "logarithmic"
    PARABOLA = "parabola"


SPLICE_METHOD_WEIGHTS: Dict[SpliceMethod, float] = {
    SpliceMethod.CUT_PASTE: 0.10,
    SpliceMethod.OLA_HANNING: 0.20,
    SpliceMethod.LINEAR: 0.15,
    SpliceMethod.COSINE: 0.20,
    SpliceMethod.HALF_SINE: 0.15,
    SpliceMethod.LOGARITHMIC: 0.10,
    SpliceMethod.PARABOLA: 0.10,
}
