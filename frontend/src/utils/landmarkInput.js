/**
 * Exact JS twin of Python's build_landmark_input from training/hand_landmarks.py
 * 
 * We write into REUSED typed arrays for maximum performance because this function
 * will run many times per second during live inference (e.g. 30 FPS).
 */

// Pre-allocate the arrays so we don't garbage collect them constantly
const landmarksOut = new Float32Array(2 * 21 * 3);
const handMask = new Float32Array(2);

export function buildLandmarkInput(handLandmarksList, frameWidth, frameHeight) {
    // Zero out the arrays from previous frames
    landmarksOut.fill(0);
    handMask.fill(0);

    if (!handLandmarksList || handLandmarksList.length === 0) {
        return { landmarks: landmarksOut, handMask, slots: 0 };
    }

    const handsPx = [];
    for (const handLms of handLandmarksList) {
        const handPx = [];
        for (let i = 0; i < handLms.length; i++) {
            const lm = handLms[i];
            handPx.push([
                lm.x * frameWidth,
                lm.y * frameHeight,
                lm.z * frameWidth
            ]);
        }
        handsPx.push(handPx);
    }

    // Sort hands by wrist (landmark 0) x_px, then y_px (tie-break rule matching Python)
    handsPx.sort((a, b) => {
        if (a[0][0] !== b[0][0]) return a[0][0] - b[0][0];
        return a[0][1] - b[0][1];
    });

    // Fill arrays (at most 2 hands)
    const numHands = Math.min(handsPx.length, 2);
    for (let i = 0; i < numHands; i++) {
        const handPx = handsPx[i];
        for (let j = 0; j < 21; j++) {
            landmarksOut[i * 21 * 3 + j * 3 + 0] = handPx[j][0];
            landmarksOut[i * 21 * 3 + j * 3 + 1] = handPx[j][1];
            landmarksOut[i * 21 * 3 + j * 3 + 2] = handPx[j][2];
        }
        handMask[i] = 1.0;
    }

    return { landmarks: landmarksOut, handMask, slots: numHands };
}
