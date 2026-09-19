import { useState, useEffect, useRef } from 'react';
import { initHandDetector, detectHands } from '../utils/handDetection';
import { buildLandmarkInput } from '../utils/landmarkInput';
import { computeCropBox } from '../utils/handCrop';
import { loadModel, predict } from '../utils/modelInference';
import { loadLabels, getLabels, getClassIndices, isSpecialClass } from '../utils/labels';

const CONFIDENCE_THRESHOLD = 0.6;
// There is no "nothing" or "space" class in the model (it only knows A-Z, 1-9).
// Therefore, when a user shows a random hand pose, the model is forced to guess.
// A slightly higher threshold (0.6) helps reject these random poses as "uncertain",
// preventing garbage predictions from bubbling up.

const SMOOTHING_WINDOW = 5;
// We average the last 5 probability vectors before making a decision.
// Why? Frame-by-frame predictions can jitter (e.g. flickering between M and N).
// Averaging over a short window (~333ms at 15fps) smooths out this noise 
// significantly, resulting in a much more stable UI experience.

export function useSignPrediction(videoRef, overlayRef, classFilter, enabled) {
    const [status, setStatus] = useState('loading'); // "loading" | "no-hand" | "uncertain" | "confident"
    const [label, setLabel] = useState('');
    const [confidence, setConfidence] = useState(0);
    const [top3, setTop3] = useState([]);
    const [handsCount, setHandsCount] = useState(0);
    const [detectionFps, setDetectionFps] = useState(0);
    const [predictionFps, setPredictionFps] = useState(0);
    const [modelLoaded, setModelLoaded] = useState(false);
    const [loadProgress, setLoadProgress] = useState(0);
    const [cropBox, setCropBox] = useState(null);
    const [lastInput, setLastInput] = useState(null);

    // Refs for animation and loop state
    const rafIdRef = useRef(null);
    const isPredictingRef = useRef(false);
    const smoothingBufferRef = useRef([]);
    const latestInputRef = useRef(null);

    // Load dependencies once
    useEffect(() => {
        let isMounted = true;
        async function init() {
            try {
                await Promise.all([
                    initHandDetector(),
                    loadLabels(),
                    loadModel((progress) => {
                        if (isMounted) setLoadProgress(progress);
                    })
                ]);
                if (isMounted) {
                    setModelLoaded(true);
                    setStatus('no-hand');
                }
            } catch (error) {
                console.error("Failed to initialize ML models:", error);
            }
        }
        init();
        return () => { isMounted = false; };
    }, []);

    // Main Loop
    useEffect(() => {
        if (!enabled || !modelLoaded || !videoRef.current || !overlayRef.current) return;
        const video = videoRef.current;
        if (video.readyState < 2) return;

        const canvas = overlayRef.current;
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        let frameCount = 0;
        let lastFpsTime = performance.now();
        
        let predCount = 0;
        let lastPredFpsTime = performance.now();

        // The Prediction Loop (Classification)
        // We run classification throttled at ~15 FPS in a separate async interval.
        // Why? 
        // 1. Detection + drawing (60 FPS) needs to be buttery smooth to track the hand visually without lag.
        // 2. Classification involves CPU compute. Even if it's fast (1-3ms), running it 60 times a second 
        //    wastes battery and compute power. 15 predictions per second is plenty for sign language.
        const predInterval = setInterval(async () => {
            if (isPredictingRef.current || !latestInputRef.current) return;
            isPredictingRef.current = true;

            const inputData = latestInputRef.current;
            const labels = getLabels();

            try {
                const { probabilities } = await predict(inputData);
                
                // Track prediction FPS
                predCount++;
                const now = performance.now();
                if (now - lastPredFpsTime >= 1000) {
                    setPredictionFps(Math.round((predCount * 1000) / (now - lastPredFpsTime)));
                    predCount = 0;
                    lastPredFpsTime = now;
                }

                // 1. Class Filter
                // Similar letter/number shapes (e.g. V vs 2, O vs 0) confuse each other.
                // Since the user usually knows whether they're spelling a word or a number,
                // we zero out the disallowed classes and renormalize to 1.0.
                const allowedIndices = getClassIndices(classFilter);
                let filteredProbs = new Float32Array(probabilities.length);
                let sum = 0;
                for (let i = 0; i < allowedIndices.length; i++) {
                    const idx = allowedIndices[i];
                    filteredProbs[idx] = probabilities[idx];
                    sum += probabilities[idx];
                }
                if (sum > 0) {
                    for (let i = 0; i < filteredProbs.length; i++) {
                        filteredProbs[i] /= sum;
                    }
                }

                // 2. Smoothing
                smoothingBufferRef.current.push(filteredProbs);
                if (smoothingBufferRef.current.length > SMOOTHING_WINDOW) {
                    smoothingBufferRef.current.shift();
                }

                const avgProbs = new Float32Array(probabilities.length);
                const bufferLen = smoothingBufferRef.current.length;
                for (const probs of smoothingBufferRef.current) {
                    for (let i = 0; i < avgProbs.length; i++) {
                        avgProbs[i] += probs[i];
                    }
                }
                for (let i = 0; i < avgProbs.length; i++) {
                    avgProbs[i] /= bufferLen;
                }

                // 3. Find top 3
                const indexedProbs = Array.from(avgProbs).map((p, i) => ({ prob: p, index: i }));
                indexedProbs.sort((a, b) => b.prob - a.prob);
                
                const currentTop3 = indexedProbs.slice(0, 3).map(x => ({
                    label: labels[x.index],
                    confidence: x.prob
                }));

                const topPick = currentTop3[0];
                
                setTop3(currentTop3);
                setLabel(topPick.label);
                setConfidence(topPick.confidence);

                if (isSpecialClass(topPick.label)) {
                    setStatus('uncertain');
                } else if (topPick.confidence >= CONFIDENCE_THRESHOLD) {
                    setStatus('confident');
                } else {
                    setStatus('uncertain');
                }

            } catch (err) {
                console.error("Prediction error:", err);
            } finally {
                isPredictingRef.current = false;
            }

        }, 1000 / 15); // ~15 FPS

        // The Detection Loop (Animation Frame)
        const loop = () => {
            const now = performance.now();
            
            frameCount++;
            if (now - lastFpsTime >= 1000) {
                setDetectionFps(Math.round((frameCount * 1000) / (now - lastFpsTime)));
                frameCount = 0;
                lastFpsTime = now;
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const results = detectHands(video, now);

            if (results) {
                const inputData = buildLandmarkInput(results.rawLandmarks, results.frameWidth, results.frameHeight);
                latestInputRef.current = inputData;
                setLastInput({ raw: results.rawLandmarks, handedness: results.handedness, built: inputData });
                setHandsCount(inputData.slots);

                // Draw skeleton (unmirrored data, onto mirrored canvas)
                const colors = ['#EE456B', '#6B5A5E']; // slot-0, slot-1
                results.rawLandmarks.forEach((hand, idx) => {
                    const color = colors[idx] || colors[0];
                    ctx.fillStyle = color;
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1.5;

                    hand.forEach(lm => {
                        ctx.beginPath();
                        ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
                        ctx.fill();
                    });

                    const HAND_CONNECTIONS = [
                        [0,1], [1,2], [2,3], [3,4],
                        [0,5], [5,6], [6,7], [7,8],
                        [5,9], [9,10], [10,11], [11,12],
                        [9,13], [13,14], [14,15], [15,16],
                        [13,17], [17,18], [18,19], [19,20],
                        [0,17]
                    ];
                    
                    ctx.beginPath();
                    HAND_CONNECTIONS.forEach(([start, end]) => {
                        ctx.moveTo(hand[start].x * canvas.width, hand[start].y * canvas.height);
                        ctx.lineTo(hand[end].x * canvas.width, hand[end].y * canvas.height);
                    });
                    ctx.stroke();
                });

                // Compute and draw crop box
                const allPx = results.rawLandmarks.flatMap(hand => 
                    hand.map(lm => [lm.x * canvas.width, lm.y * canvas.height])
                );
                const box = computeCropBox(allPx, canvas.width, canvas.height);
                setCropBox(box);

                ctx.strokeStyle = '#EE456B';
                ctx.lineWidth = 2;
                ctx.strokeRect(box.x0, box.y0, box.side, box.side);

            } else {
                // No hands detected
                latestInputRef.current = null;
                smoothingBufferRef.current = [];
                setStatus('no-hand');
                setLabel('');
                setConfidence(0);
                setTop3([]);
                setHandsCount(0);
                setCropBox(null);
                setLastInput(null);
            }

            rafIdRef.current = requestAnimationFrame(loop);
        };

        rafIdRef.current = requestAnimationFrame(loop);

        return () => {
            if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
            clearInterval(predInterval);
            // reset state on unmount/disable
            smoothingBufferRef.current = [];
            latestInputRef.current = null;
            setStatus(modelLoaded ? 'no-hand' : 'loading');
        };
    }, [enabled, modelLoaded, classFilter, videoRef, overlayRef]);

    return {
        status, label, confidence, top3, handsCount, 
        detectionFps, predictionFps, modelLoaded, loadProgress, 
        cropBox, lastInput
    };
}
