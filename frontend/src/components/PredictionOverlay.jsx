import React from 'react';

export default function PredictionOverlay({ 
    status, 
    label, 
    confidence, 
    top3, 
    handsCount, 
    loadProgress 
}) {
    // Determine opacity/fade based on status
    const isFaded = status === 'uncertain';
    
    return (
        <div className="bg-cream-50 rounded-2xl border border-cream-300 shadow-[0_4px_24px_rgba(46,34,38,0.05)] p-6 md:p-8 flex flex-col h-full min-h-[320px] transition-all duration-300">
            
            {status === 'loading' ? (
                <div className="flex-1 flex flex-col items-center justify-center gap-4">
                    <p className="text-ink-muted text-lg">Loading model…</p>
                    <div className="w-full max-w-[200px] h-2 bg-blush-100 rounded-full overflow-hidden">
                        <div 
                            className="h-full bg-blush-500 transition-all duration-300" 
                            style={{ width: `${loadProgress}%` }}
                        />
                    </div>
                </div>
            ) : status === 'no-hand' ? (
                <div className="flex-1 flex items-center justify-center">
                    <p className="text-ink-muted text-xl text-center">Show your hands to the camera</p>
                </div>
            ) : (
                <div className="flex-1 flex flex-col items-center justify-center relative">
                    
                    {/* Status Text (Uncertain) */}
                    <div className="absolute top-0 text-center w-full h-6">
                        {isFaded && (
                            <span className="text-ink-muted text-sm font-medium animate-pulse">
                                Not sure yet
                            </span>
                        )}
                    </div>

                    {/* Main Prediction */}
                    <div className={`transition-opacity duration-300 ${isFaded ? 'opacity-50' : 'opacity-100'} text-center mt-6`}>
                        <h2 className="font-display text-8xl md:text-[9rem] text-blush-800 leading-none">
                            {label || '-'}
                        </h2>
                    </div>

                    {/* Confidence & Hands Info */}
                    <div className="w-full mt-auto pt-6 flex flex-col gap-4">
                        <div className="flex justify-between items-end mb-1">
                            <span className="text-sm font-semibold text-blush-800">
                                Confidence
                            </span>
                            <span className="text-sm text-ink-muted">
                                {Math.round(confidence * 100)}%
                            </span>
                        </div>
                        <div className="w-full h-3 bg-blush-100 rounded-full overflow-hidden">
                            <div 
                                className="h-full bg-blush-600 rounded-full transition-all duration-150 ease-out"
                                style={{ width: `${Math.round(confidence * 100)}%` }}
                            />
                        </div>

                        <div className="flex justify-between items-center mt-2">
                            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                                handsCount > 0 ? 'bg-blush-100 text-blush-800' : 'bg-cream-200 text-ink-muted'
                            }`}>
                                {handsCount} {handsCount === 1 ? 'hand' : 'hands'}
                            </span>
                        </div>

                        {/* Top 3 List (for demoing confusions) */}
                        {top3 && top3.length > 1 && (
                            <div className="mt-4 pt-4 border-t border-cream-300">
                                <p className="text-xs text-ink-muted mb-2 uppercase tracking-wide">Top Alternates</p>
                                <div className="flex gap-4">
                                    {top3.slice(1).map((alt, idx) => (
                                        <div key={idx} className="flex flex-col">
                                            <span className="text-sm font-display text-ink font-semibold">{alt.label}</span>
                                            <span className="text-xs text-ink-muted">{Math.round(alt.confidence * 100)}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
