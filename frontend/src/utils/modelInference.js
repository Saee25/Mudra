import * as ort from 'onnxruntime-web';

// Set ORT environment variables
ort.env.wasm.wasmPaths = import.meta.env.BASE_URL + "ort/";
// A tiny MLP doesn't need threads, and without COOP/COEP headers, SharedArrayBuffer isn't available anyway.
ort.env.wasm.numThreads = 1;

let session = null;

// Reusable buffers
const landmarksBuffer = new Float32Array(1 * 2 * 21 * 3);
const handMaskBuffer = new Float32Array(1 * 2);

/* 
 * PREPROCESSING PARITY TABLE (Python Training vs. Browser)
 * 
 * | Feature                  | Python (Training)                     | Browser (Inference)                   | Match? |
 * |--------------------------|---------------------------------------|---------------------------------------|--------|
 * | Hand Model               | hand_landmarker.task                  | hand_landmarker.task                  | YES    |
 * | Num Hands                | 2                                     | 2                                     | YES    |
 * | Confidences              | 0.3 (detect), 0.3 (presence)          | 0.3 (detect), 0.3 (presence)          | YES    |
 * | Running Mode             | IMAGE                                 | VIDEO (with tracking)                 | MINOR* |
 * | Frames                   | Unmirrored                            | Unmirrored                            | YES    |
 * | Color                    | RGB (from BGR)                        | RGB (from webcam)                     | YES    |
 * | Coordinate Conversion    | x*W, y*H, z*W                         | x*W, y*H, z*W                         | YES    |
 * | Slot Ordering            | wrist x (smallest first)              | wrist x (smallest first)              | YES    |
 * | Missing Hand Handling    | Zeros + mask 0                        | Zeros + mask 0                        | YES    |
 * | Normalization            | Inside the model                      | Inside the model                      | YES    |
 * | Softmax                  | Inside the model                      | Inside the model                      | YES    |
 * | Dtype and Shapes         | float32, [1,2,21,3] + [1,2]           | float32, [1,2,21,3] + [1,2]           | YES    |
 * | Class Filter             | N/A (Training sees all)               | Applied AFTER the model output        | YES    |
 * 
 * *MINOR DIFFERENCE: Training uses IMAGE mode (independent frames) while the browser uses VIDEO mode 
 * (tracks hands across frames). This is acceptable as VIDEO mode is more stable for live input.
 */

export async function loadModel(onProgress) {
    if (session) return session;

    console.log("Creating ORT session...");
    const modelUrl = import.meta.env.BASE_URL + "model/mudra.onnx";
    
    // Pass the URL directly so ORT can resolve mudra.onnx.data relative to it
    session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    console.log(`Session created.`);

    if (onProgress) {
        onProgress(100);
    }

    // Warm-up run: The first run of a WASM model is usually slower due to WebAssembly 
    // JIT compilation and initialization. We run it once with zeros to get it out of the way.
    landmarksBuffer.fill(0);
    handMaskBuffer.fill(0);
    const warmupLandmarksTensor = new ort.Tensor('float32', landmarksBuffer, [1, 2, 21, 3]);
    const warmupMaskTensor = new ort.Tensor('float32', handMaskBuffer, [1, 2]);
    await session.run({ landmarks: warmupLandmarksTensor, hand_mask: warmupMaskTensor });
    console.log("Model warm-up complete.");

    return session;
}

export async function predict(inputData) {
    if (!session) throw new Error("Model not loaded");

    // Copy data into our reusable buffers
    landmarksBuffer.set(inputData.landmarks);
    handMaskBuffer.set(inputData.handMask);

    const landmarksTensor = new ort.Tensor('float32', landmarksBuffer, [1, 2, 21, 3]);
    const maskTensor = new ort.Tensor('float32', handMaskBuffer, [1, 2]);

    const start = performance.now();
    // Feeds { landmarks: [1,2,21,3], hand_mask: [1,2] }
    // Reads "probabilities" (already softmaxed in the model)
    const results = await session.run({ landmarks: landmarksTensor, hand_mask: maskTensor });
    const inferenceMs = performance.now() - start;

    return { probabilities: results.probabilities.data, inferenceMs };
}
