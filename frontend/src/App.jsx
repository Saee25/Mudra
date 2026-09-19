import React, { useEffect, useRef, useState } from 'react';
import Webcam from './components/Webcam';
import { initHandDetector, detectHands, getActiveDelegate } from './utils/handDetection';
import { buildLandmarkInput } from './utils/landmarkInput';
import { computeCropBox, cropHands, CROP_SIZE } from './utils/handCrop';
import { runParityCheck } from './utils/parityCheck';

function App() {
    const videoRef = useRef(null);
    const overlayRef = useRef(null);
    const cropPreviewRef = useRef(null);
    const rafIdRef = useRef(null);

    const [isDetectorReady, setIsDetectorReady] = useState(false);
    const [stats, setStats] = useState({ fps: 0, hands: 0, delegate: 'Loading...' });

    // 1. Initialize detector & parity check
    useEffect(() => {
        runParityCheck();
        
        initHandDetector().then(() => {
            setIsDetectorReady(true);
            setStats(s => ({ ...s, delegate: getActiveDelegate() }));
        }).catch(err => {
            console.error("Failed to initialize detector:", err);
            setStats(s => ({ ...s, delegate: 'Failed' }));
        });
    }, []);

    // 2. Main inference loop
    const handleWebcamReady = () => {
        if (!isDetectorReady) return;

        const video = videoRef.current;
        const canvas = overlayRef.current;
        const ctx = canvas.getContext('2d');
        
        const previewCanvas = cropPreviewRef.current;
        const previewCtx = previewCanvas?.getContext('2d');

        // Match canvas size to video size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        let lastTime = performance.now();
        let frameCount = 0;
        let lastFpsTime = lastTime;

        const loop = () => {
            const now = performance.now();
            
            // FPS calculation
            frameCount++;
            if (now - lastFpsTime >= 1000) {
                setStats(s => ({ ...s, fps: Math.round((frameCount * 1000) / (now - lastFpsTime)) }));
                frameCount = 0;
                lastFpsTime = now;
            }

            // Clear overlay
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // MediaPipe requires strictly increasing timestamps for VIDEO mode
            // We use performance.now() as it satisfies this.
            const results = detectHands(video, now);

            if (results) {
                // 1. JS Parity: Build input features (we don't use the output yet, just verify it runs fast)
                const inputData = buildLandmarkInput(results.rawLandmarks, results.frameWidth, results.frameHeight);
                setStats(s => ({ ...s, hands: inputData.slots }));

                // 2. Draw landmarks
                // We use the normalized coordinates for drawing to match the canvas dimensions easily
                const colors = ['#EE456B', '#6B5A5E']; // slot-0: blush-500, slot-1: ink-muted
                
                results.rawLandmarks.forEach((hand, idx) => {
                    const color = colors[idx] || colors[0];
                    ctx.fillStyle = color;
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1.5;

                    // Draw dots
                    hand.forEach(lm => {
                        ctx.beginPath();
                        // IMPORTANT: The canvas is mirrored via CSS, so we just draw using 
                        // original x,y and it aligns visually!
                        ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
                        ctx.fill();
                    });

                    // Draw connections (skeleton)
                    const HAND_CONNECTIONS = [
                        [0,1], [1,2], [2,3], [3,4], // thumb
                        [0,5], [5,6], [6,7], [7,8], // index
                        [5,9], [9,10], [10,11], [11,12], // middle
                        [9,13], [13,14], [14,15], [15,16], // ring
                        [13,17], [17,18], [18,19], [19,20], // pinky
                        [0,17] // palm
                    ];
                    
                    ctx.beginPath();
                    HAND_CONNECTIONS.forEach(([start, end]) => {
                        const pt1 = hand[start];
                        const pt2 = hand[end];
                        ctx.moveTo(pt1.x * canvas.width, pt1.y * canvas.height);
                        ctx.lineTo(pt2.x * canvas.width, pt2.y * canvas.height);
                    });
                    ctx.stroke();
                });

                // 3. Draw crop box
                // Convert normalized landmarks to pixel landmarks for the parity function
                const allPx = results.rawLandmarks.flatMap(hand => 
                    hand.map(lm => [lm.x * canvas.width, lm.y * canvas.height])
                );
                const box = computeCropBox(allPx, canvas.width, canvas.height);
                
                ctx.strokeStyle = '#EE456B'; // blush-500
                ctx.lineWidth = 2;
                ctx.strokeRect(box.x0, box.y0, box.side, box.side);

                // 4. Update live crop preview
                if (previewCtx) {
                    const cropCanvas = cropHands(video, box);
                    previewCtx.clearRect(0, 0, CROP_SIZE, CROP_SIZE);
                    // Mirror the preview canvas as well (via drawing with negative scale)
                    // so it matches the mirrored UI.
                    previewCtx.save();
                    previewCtx.scale(-1, 1);
                    previewCtx.drawImage(cropCanvas, -CROP_SIZE, 0);
                    previewCtx.restore();
                }

            } else {
                setStats(s => ({ ...s, hands: 0 }));
                if (previewCtx) previewCtx.clearRect(0, 0, CROP_SIZE, CROP_SIZE);
            }

            rafIdRef.current = requestAnimationFrame(loop);
        };

        loop();
    };

    // If detector finishes loading after webcam is ready, trigger loop
    useEffect(() => {
        if (isDetectorReady && videoRef.current && videoRef.current.readyState >= 2) {
            handleWebcamReady();
        }
        return () => {
            if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
        };
    }, [isDetectorReady]);

    return (
        <div className="min-h-screen bg-cream-100 text-ink font-sans flex flex-col items-center py-8 px-4">
            
            {/* Header */}
            <header className="mb-8 text-center">
                <h1 className="text-4xl md:text-5xl font-display font-semibold text-blush-700 tracking-tight">Mudra</h1>
                <p className="text-ink-muted mt-2">Indian Sign Language Recognizer</p>
            </header>

            <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Main Camera Area */}
                <div className="lg:col-span-2 relative aspect-[4/3] w-full max-w-3xl mx-auto">
                    <Webcam 
                        ref={videoRef} 
                        onReady={handleWebcamReady} 
                        className="w-full h-full absolute inset-0"
                    />
                    
                    {/* 
                      * OVERLAY CANVAS
                      * Must perfectly overlay the video and apply the exact same CSS mirroring.
                      */}
                    <canvas 
                        ref={overlayRef}
                        className="absolute inset-0 w-full h-full pointer-events-none -scale-x-100"
                    />
                </div>

                {/* Debug / Info Card */}
                <div className="flex flex-col gap-6">
                    <div className="bg-cream-50 rounded-2xl border border-cream-300 shadow-[0_4px_24px_rgba(46,34,38,0.05)] p-6">
                        <h2 className="font-display text-xl text-blush-800 mb-4">Debug Info</h2>
                        
                        <div className="space-y-4 text-sm">
                            <div className="flex justify-between items-center border-b border-cream-300 pb-2">
                                <span className="text-ink-muted">FPS</span>
                                <span className="font-medium">{stats.fps}</span>
                            </div>
                            
                            <div className="flex justify-between items-center border-b border-cream-300 pb-2">
                                <span className="text-ink-muted">Active Delegate</span>
                                <span className="font-medium">{stats.delegate}</span>
                            </div>
                            
                            <div className="flex justify-between items-center border-b border-cream-300 pb-2">
                                <span className="text-ink-muted">Detected Hands</span>
                                <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                                    stats.hands > 0 
                                    ? 'bg-blush-100 text-blush-800' 
                                    : 'bg-cream-200 text-ink-muted'
                                }`}>
                                    {stats.hands} / 2
                                </span>
                            </div>

                            <div className="pt-2 text-xs text-ink-muted">
                                <p><strong>Slot Guide:</strong></p>
                                <ul className="list-disc list-inside mt-1 space-y-1">
                                    <li><span className="inline-block w-2 h-2 rounded-full bg-[#EE456B] mr-1"></span> Slot 0 (left-most)</li>
                                    <li><span className="inline-block w-2 h-2 rounded-full bg-[#6B5A5E] mr-1"></span> Slot 1 (right-most)</li>
                                </ul>
                            </div>
                        </div>
                    </div>

                    {/* Live Crop Preview */}
                    <div className="bg-cream-50 rounded-2xl border border-cream-300 shadow-[0_4px_24px_rgba(46,34,38,0.05)] p-6 flex flex-col items-center">
                        <h2 className="font-display text-xl text-blush-800 mb-4 w-full text-left">Crop Preview</h2>
                        <div className="w-[224px] h-[224px] bg-cream-100 rounded-lg overflow-hidden border border-cream-300 relative">
                            <canvas 
                                ref={cropPreviewRef} 
                                width={CROP_SIZE} 
                                height={CROP_SIZE}
                                className="w-full h-full"
                            />
                            {stats.hands === 0 && (
                                <div className="absolute inset-0 flex items-center justify-center text-ink-muted text-sm">
                                    No hands detected
                                </div>
                            )}
                        </div>
                        <p className="text-xs text-ink-muted text-center mt-3">
                            224x224 crop used for comparison models and data collection. The deployed model uses landmark points directly.
                        </p>
                    </div>
                </div>

            </div>
        </div>
    );
}

export default App;
