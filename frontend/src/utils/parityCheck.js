import { buildLandmarkInput } from './landmarkInput.js';
import { computeCropBox } from './handCrop.js';
import { loadModel, predict } from './modelInference.js';

/**
 * Dev-only unit test runner to verify JS preprocessing perfectly matches Python.
 */
export async function runParityCheck() {
    if (!import.meta.env.DEV) return;
    
    console.log('Running preprocessing parity checks...');
    
    try {
        const response = await fetch(`${import.meta.env.BASE_URL}model/parity_unit_cases.json`);
        if (!response.ok) {
            console.warn('parity_unit_cases.json not found, skipping parity check.');
            return;
        }
        const unitCases = await response.json();
        
        const results = [];
        let allPassed = true;
        
        // Helper to check arrays
        const arraysClose = (a, b, tol=1e-4) => {
            if (a.length !== b.length) return false;
            for (let i=0; i<a.length; i++) {
                if (Math.abs(a[i] - b[i]) > tol) return false;
            }
            return true;
        };

        // 1. One hand
        if (unitCases.one_hand) {
            const h1 = Array.from({length: 21}, (_, i) => ({x: 0.1 + i*0.001, y: 0.2, z: 0.3}));
            const jsOut = buildLandmarkInput([h1], 1000, 1000);
            
            const expectedLms = unitCases.one_hand.landmarks.flat(2); // flatten 2D array
            const passedLms = arraysClose(jsOut.landmarks, expectedLms);
            const passedMask = arraysClose(jsOut.handMask, unitCases.one_hand.mask);
            
            results.push({
                Test: 'one_hand',
                Passed: passedLms && passedMask ? '✅' : '❌'
            });
            if (!(passedLms && passedMask)) allPassed = false;
        }

        // 2. Two hands order
        if (unitCases.two_hands_order) {
            const h1 = Array.from({length: 21}, (_, i) => ({x: 0.1 + i*0.001, y: 0.2, z: 0.3}));
            const h2 = Array.from({length: 21}, (_, i) => ({x: 0.05 + i*0.001, y: 0.2, z: 0.3}));
            const jsOut = buildLandmarkInput([h1, h2], 1000, 1000);
            
            const expectedLms = unitCases.two_hands_order.landmarks.flat(2);
            const passedLms = arraysClose(jsOut.landmarks, expectedLms);
            const passedMask = arraysClose(jsOut.handMask, unitCases.two_hands_order.mask);
            
            results.push({
                Test: 'two_hands_order',
                Passed: passedLms && passedMask ? '✅' : '❌'
            });
            if (!(passedLms && passedMask)) allPassed = false;
        }

        // 3. Crop Box Edge
        if (unitCases.box_edge) {
            const pts = [[5, 5, 0], [15, 15, 0]];
            const jsOut = computeCropBox(pts, 100, 100);
            const passed = jsOut.x0 === unitCases.box_edge.x0 && 
                           jsOut.y0 === unitCases.box_edge.y0 && 
                           jsOut.side === unitCases.box_edge.side;
            results.push({ Test: 'box_edge', Passed: passed ? '✅' : '❌' });
            if (!passed) allPassed = false;
        }

        // 4. Crop Box Large
        if (unitCases.box_large) {
            const pts = [[10, 10, 0], [90, 90, 0]];
            const jsOut = computeCropBox(pts, 50, 50);
            const passed = jsOut.x0 === unitCases.box_large.x0 && 
                           jsOut.y0 === unitCases.box_large.y0 && 
                           jsOut.side === unitCases.box_large.side;
            results.push({ Test: 'box_large', Passed: passed ? '✅' : '❌' });
            if (!passed) allPassed = false;
        }

        // 5. Crop Box Round
        if (unitCases.box_round) {
            const pts = [[0, 0, 0], [9, 9, 0]];
            const jsOut = computeCropBox(pts, 100, 100);
            const passed = jsOut.x0 === unitCases.box_round.x0 && 
                           jsOut.y0 === unitCases.box_round.y0 && 
                           jsOut.side === unitCases.box_round.side;
            results.push({ Test: 'box_round', Passed: passed ? '✅' : '❌' });
            if (!passed) allPassed = false;
        }

        console.table(results);
        
        // ------------- ONNX Model Parity Check -------------
        console.log('Running ONNX model parity checks...');
        const samplesRes = await fetch(`${import.meta.env.BASE_URL}model/parity_samples.json`);
        if (!samplesRes.ok) {
            console.warn('parity_samples.json not found, skipping full model parity check.');
            return;
        }
        const paritySamples = await samplesRes.json();
        
        await loadModel();
        
        const modelResults = [];
        for (const sample of paritySamples) {
            // Reconstruct the nested mediapipe-like output
            // Each sample.raw_landmarks is an array of dicts {x, y, z}.
            // Note: the training script exports it as an array of hands.
            const jsInput = buildLandmarkInput(sample.raw_landmarks, sample.frame_width, sample.frame_height);
            
            const { probabilities } = await predict(jsInput);
            
            let maxDiff = 0;
            for (let i = 0; i < probabilities.length; i++) {
                const diff = Math.abs(probabilities[i] - sample.py_probs[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            
            // Top 1 agreement
            let jsTop1Idx = 0, pyTop1Idx = 0;
            for (let i = 1; i < probabilities.length; i++) {
                if (probabilities[i] > probabilities[jsTop1Idx]) jsTop1Idx = i;
                if (sample.py_probs[i] > sample.py_probs[pyTop1Idx]) pyTop1Idx = i;
            }
            const top1Match = jsTop1Idx === pyTop1Idx;
            
            const passed = maxDiff < 1e-4 && top1Match;
            if (!passed) allPassed = false;
            
            modelResults.push({
                File: sample.file.substring(0, 20) + '...',
                MaxDiff: maxDiff.toExponential(2),
                Top1Match: top1Match ? '✅' : '❌'
            });
        }
        
        console.table(modelResults);

        if (allPassed) {
            console.log('%cMudra parity check: PASS', 'color: green; font-weight: bold;');
        } else {
            console.error('Mudra parity check: FAIL! Check console table for details.');
        }

    } catch (err) {
        console.error('Error running parity check:', err);
    }
}
