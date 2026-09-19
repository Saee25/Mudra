import React, { useEffect, useRef, useState } from 'react';
import Webcam from './components/Webcam';
import PredictionOverlay from './components/PredictionOverlay';
import { useSignPrediction } from './hooks/useSignPrediction';
import { runParityCheck } from './utils/parityCheck';
import { getActiveDelegate } from './utils/handDetection';

function App() {
    const videoRef = useRef(null);
    const overlayRef = useRef(null);

    // Hardcode class filter to 'both' for now, can be a toggle later
    const [classFilter, setClassFilter] = useState('both'); 
    
    // Use our new hook!
    const { 
        status, label, confidence, top3, handsCount, 
        detectionFps, predictionFps, modelLoaded, loadProgress, 
        cropBox 
    } = useSignPrediction(videoRef, overlayRef, classFilter, true);

    // 1. Initialize parity check on mount
    useEffect(() => {
        runParityCheck();
    }, []);

    return (
        <div className="min-h-screen bg-cream-100 text-ink font-sans flex flex-col items-center py-8 px-4">
            
            {/* Header */}
            <header className="mb-8 text-center">
                <h1 className="text-4xl md:text-5xl font-display font-semibold text-blush-700 tracking-tight">Mudra</h1>
                <p className="text-ink-muted mt-2">Indian Sign Language Recognizer</p>
            </header>

            <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Main Camera Area */}
                <div className="lg:col-span-2 relative aspect-[4/3] w-full mx-auto bg-cream-50 rounded-2xl overflow-hidden shadow-[0_4px_24px_rgba(46,34,38,0.05)] border border-cream-300">
                    <Webcam 
                        ref={videoRef} 
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
                    
                    {/* Mini FPS/Debug Indicator */}
                    <div className="absolute top-4 left-4 bg-cream-50/80 backdrop-blur-md px-3 py-1.5 rounded-full border border-cream-300 shadow-sm text-xs text-ink-muted flex items-center gap-3">
                        <span>Det: {detectionFps} FPS</span>
                        <span>Pred: {predictionFps} FPS</span>
                        <span>{getActiveDelegate() || 'CPU'}</span>
                    </div>
                </div>

                {/* Prediction Overlay Card */}
                <div className="lg:col-span-1">
                    <PredictionOverlay 
                        status={status}
                        label={label}
                        confidence={confidence}
                        top3={top3}
                        handsCount={handsCount}
                        loadProgress={loadProgress}
                    />
                </div>
            </div>

        </div>
    );
}

export default App;
