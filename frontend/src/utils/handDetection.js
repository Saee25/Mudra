import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

let detectorInstance = null;
let detectorPromise = null;
let activeDelegate = 'CPU'; // We will log which one we got

export async function initHandDetector() {
    if (detectorPromise) return detectorPromise;
    
    detectorPromise = (async () => {
        try {
            console.log('Loading MediaPipe HandLandmarker WASM...');
            const vision = await FilesetResolver.forVisionTasks(import.meta.env.BASE_URL + "mediapipe/wasm");
            
            // These confidence values must match `training/hand_landmarks.py`.
            // A threshold of 0.3 improves two-hand recall when hands overlap.
            const minDetectionConfidence = 0.3;
            const minPresenceConfidence = 0.3;
            
            const options = {
                baseOptions: {
                    modelAssetPath: import.meta.env.BASE_URL + "mediapipe/hand_landmarker.task",
                    delegate: "GPU" // Try GPU first
                },
                runningMode: "VIDEO",
                numHands: 2,
                minHandDetectionConfidence: minDetectionConfidence,
                minHandPresenceConfidence: minPresenceConfidence
            };

            let landmarker;
            try {
                console.log('Attempting to initialize HandLandmarker with GPU delegate...');
                landmarker = await HandLandmarker.createFromOptions(vision, options);
                activeDelegate = 'GPU';
            } catch (err) {
                console.warn('GPU delegate failed or unavailable. Falling back to CPU.', err);
                options.baseOptions.delegate = "CPU";
                landmarker = await HandLandmarker.createFromOptions(vision, options);
                activeDelegate = 'CPU';
            }
            
            console.log(`HandLandmarker initialized successfully (Delegate: ${activeDelegate}).`);
            detectorInstance = landmarker;
            return landmarker;
        } catch (error) {
            console.error('Failed to initialize HandLandmarker:', error);
            detectorPromise = null; // allow retrying
            throw error;
        }
    })();

    return detectorPromise;
}

export function getActiveDelegate() {
    return activeDelegate;
}

export function detectHands(videoElement, timestampMs) {
    if (!detectorInstance) {
        throw new Error('HandLandmarker not initialized. Call initHandDetector first.');
    }
    
    // VIDEO mode requires a strictly increasing timestamp for tracking across frames.
    const results = detectorInstance.detectForVideo(videoElement, timestampMs);
    
    if (results.landmarks && results.landmarks.length > 0) {
        return {
            rawLandmarks: results.landmarks,
            handedness: results.handedness,
            frameWidth: videoElement.videoWidth,
            frameHeight: videoElement.videoHeight
        };
    }
    
    return null;
}
