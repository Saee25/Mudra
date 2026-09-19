// Motion signs that cannot be fairly judged by a single-frame static model.
// TODO: Verify with the official ISLRTC alphabet chart and add any motion-based letters (e.g. J, Z in some sign languages, but ISL may be different).
export const MOTION_SIGNS = [];

// Hints for Practice Mode. 
// ONLY descriptions verified against the Indian Sign Language Research and Training Centre (ISLRTC) 
// should be hardcoded here.
//
// TODO(verify with ISLRTC): We are using the neutral fallback for all signs 
// until they are officially verified to avoid hallucinating incorrect hand shapes.
const FALLBACK_HINT = "Compare with the ISLRTC reference chart.";

export const SIGN_HINTS = {
    // Alphabet
    "A": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "B": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "C": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "D": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "E": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "F": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "G": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "H": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "I": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "J": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "K": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "L": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "M": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "N": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "O": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "P": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "Q": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "R": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "S": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "T": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "U": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "V": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "W": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "X": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "Y": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "Z": FALLBACK_HINT, // TODO(verify with ISLRTC)

    // Numbers
    "1": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "2": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "3": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "4": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "5": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "6": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "7": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "8": FALLBACK_HINT, // TODO(verify with ISLRTC)
    "9": FALLBACK_HINT, // TODO(verify with ISLRTC)
};
