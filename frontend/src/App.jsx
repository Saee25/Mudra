import React, { useEffect, useRef, useState } from 'react';
import Webcam from './components/Webcam';
import PredictionOverlay from './components/PredictionOverlay';
import WordBuilder from './components/WordBuilder';
import PracticeMode from './components/PracticeMode';
import { useSignPrediction } from './hooks/useSignPrediction';
import { runParityCheck } from './utils/parityCheck';
import { getActiveDelegate } from './utils/handDetection';

function App() {
    const videoRef = useRef(null);
    const overlayRef = useRef(null);

    const [classFilter, setClassFilter] = useState('letters'); 
    const [appMode, setAppMode] = useState('free'); // 'free' | 'practice'
    
    // Core ML hook
    const { 
        status, label, confidence, top3, handsCount, 
        detectionFps, predictionFps, modelLoaded, loadProgress
    } = useSignPrediction(videoRef, overlayRef, classFilter, true);

    // Initialize parity check on mount (dev only, can check console)
    useEffect(() => {
        runParityCheck();
    }, []);

    return (
        <div className="min-h-screen bg-cream-100 text-ink font-sans flex flex-col items-center py-6 px-4 md:py-10">
            
            {/* Header */}
            <header className="mb-8 text-center flex flex-col items-center">
                <h1 className="text-5xl md:text-6xl font-display font-semibold text-blush-800 tracking-tight">Mudra</h1>
                <p className="text-ink-muted mt-3 text-sm md:text-base font-medium max-w-sm">
                    Real-time Indian Sign Language recognition, right in your browser.
                </p>

                {/* Mode Toggle */}
                <div className="mt-8 flex bg-cream-200 rounded-full p-1 shadow-inner">
                    <button
                        onClick={() => setAppMode('free')}
                        className={`px-5 py-2 rounded-full text-sm font-medium transition-colors duration-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blush-600 ${
                            appMode === 'free' 
                                ? 'bg-blush-700 text-white shadow-sm' 
                                : 'text-ink-muted hover:text-ink'
                        }`}
                    >
                        Free mode
                    </button>
                    <button
                        onClick={() => setAppMode('practice')}
                        className={`px-5 py-2 rounded-full text-sm font-medium transition-colors duration-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blush-600 ${
                            appMode === 'practice' 
                                ? 'bg-blush-700 text-white shadow-sm' 
                                : 'text-ink-muted hover:text-ink'
                        }`}
                    >
                        Practice mode
                    </button>
                </div>
            </header>

            {/* Main Layout Area */}
            <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                
                {/* Left Column: Camera */}
                <div className="lg:col-span-7 flex flex-col gap-4">
                    <div className="relative aspect-[4/3] w-full bg-cream-50 rounded-2xl overflow-hidden shadow-[0_4px_24px_rgba(46,34,38,0.05)] border border-cream-300">
                        <Webcam 
                            ref={videoRef} 
                            className="w-full h-full absolute inset-0"
                        />
                        
                        {/* OVERLAY CANVAS */}
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
                </div>

                {/* Right Column: Interactive UI */}
                <div className="lg:col-span-5 flex flex-col gap-6">
                    {/* Prediction Overlay (The large current prediction) */}
                    <PredictionOverlay 
                        status={status}
                        label={label}
                        confidence={confidence}
                        top3={top3}
                        handsCount={handsCount}
                        loadProgress={loadProgress}
                    />

                    {/* Mode Specific UI */}
                    {appMode === 'free' ? (
                        <WordBuilder 
                            status={status}
                            label={label}
                            classFilter={classFilter}
                            setClassFilter={setClassFilter}
                        />
                    ) : (
                        <PracticeMode 
                            status={status}
                            label={label}
                            classFilter={classFilter}
                            setClassFilter={setClassFilter}
                        />
                    )}
                </div>
            </div>

            {/* Footer */}
            <footer className="mt-16 text-center text-xs text-ink-muted max-w-2xl px-4 flex flex-col gap-2">
                <p>
                    All processing happens in your browser &mdash; no video leaves your device.
                </p>
                <p>
                    Mudra recognizes the static ISL fingerspelling alphabet and numbers. 
                    ISL has regional variations; signs may differ from what you learned.
                </p>
            </footer>

        </div>
    );
}

export default App;
