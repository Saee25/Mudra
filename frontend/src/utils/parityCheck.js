import { buildLandmarkInput } from './landmarkInput.js';
import { computeCropBox } from './handCrop.js';

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
        
        if (allPassed) {
            console.log('%cParity Check Passed!', 'color: green; font-weight: bold;');
        } else {
            console.error('Parity Check Failed! Check console table for details.');
        }

    } catch (err) {
        console.error('Error running parity check:', err);
    }
}
