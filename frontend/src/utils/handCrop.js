export const CROP_PADDING = 0.2;
export const CROP_SIZE = 224;

/**
 * JS twin of Python's compute_crop_box from training/hand_landmarks.py
 * 
 * This is the square crop around the UNION of all detected hands' landmarks.
 * The deployed model does NOT use crops; this is only for the debug preview 
 * and Collect mode's optional crops.
 * 
 * Algorithm (Parity-Critical):
 * 1. tight box around all landmark points (both hands).
 * 2. side = max(box_w, box_h) * (1 + 2 * CROP_PADDING), centered on the box center.
 * 3. if side > min(frame_width, frame_height), set side = min(frame_width, frame_height).
 * 4. x0 = cx - side/2 and y0 = cy - side/2, then SHIFT the box back inside the frame 
 *    if it overflows an edge (keeping its size). This avoids distortion and black padding.
 * 5. rounding rule: round side, x0, and y0 to integers with Math.floor(v + 0.5) 
 *    to exactly match Python's int(math.floor(v + 0.5)). Round side first, then 
 *    compute and round x0/y0, then shift.
 */
export function computeCropBox(allLandmarksPx, frameWidth, frameHeight) {
    if (!allLandmarksPx || allLandmarksPx.length === 0) {
        return { x0: 0, y0: 0, side: Math.min(frameWidth, frameHeight) };
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (const pt of allLandmarksPx) {
        if (pt[0] < minX) minX = pt[0];
        if (pt[0] > maxX) maxX = pt[0];
        if (pt[1] < minY) minY = pt[1];
        if (pt[1] > maxY) maxY = pt[1];
    }

    const boxW = maxX - minX;
    const boxH = maxY - minY;
    const cx = minX + boxW / 2.0;
    const cy = minY + boxH / 2.0;

    let side = Math.max(boxW, boxH) * (1.0 + 2.0 * CROP_PADDING);

    // Cap side to max possible square crop
    const maxSide = Math.min(frameWidth, frameHeight);
    if (side > maxSide) {
        side = maxSide;
    }

    // Round side first
    side = Math.floor(side + 0.5);

    // Compute corner and round
    let x0 = Math.floor(cx - side / 2.0 + 0.5);
    let y0 = Math.floor(cy - side / 2.0 + 0.5);

    // Shift to keep inside frame
    if (x0 < 0) {
        x0 = 0;
    } else if (x0 + side > frameWidth) {
        x0 = frameWidth - side;
    }

    if (y0 < 0) {
        y0 = 0;
    } else if (y0 + side > frameHeight) {
        y0 = frameHeight - side;
    }

    return { x0, y0, side };
}

/**
 * Creates a reusable 224x224 offscreen canvas containing the cropped region.
 */
export function cropHands(videoElement, box) {
    const canvas = document.createElement('canvas');
    canvas.width = CROP_SIZE;
    canvas.height = CROP_SIZE;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    // Draw the specified box from the video to the 224x224 canvas
    ctx.drawImage(
        videoElement, 
        box.x0, box.y0, box.side, box.side, // Source rectangle
        0, 0, CROP_SIZE, CROP_SIZE          // Destination rectangle
    );
    
    return canvas;
}
