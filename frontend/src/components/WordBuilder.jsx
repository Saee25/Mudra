import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useHoldToCommit } from '../hooks/useHoldToCommit';
import { Copy, Volume2, Delete, Trash2, Space } from 'lucide-react';

export default function WordBuilder({ status, label, classFilter, setClassFilter }) {
    const [textBuffer, setTextBuffer] = useState('');
    const [copied, setCopied] = useState(false);
    const [canSpeak, setCanSpeak] = useState(false);
    const [lastCharCommitted, setLastCharCommitted] = useState(false);

    // Initialize Speech Synthesis availability
    useEffect(() => {
        if ('speechSynthesis' in window) {
            setCanSpeak(true);
        }
    }, []);

    const handleCommit = useCallback((newLabel) => {
        setTextBuffer(prev => prev + newLabel);
        
        // Highlight animation trigger
        setLastCharCommitted(true);
        setTimeout(() => setLastCharCommitted(false), 500);
    }, []);

    const { progress, isHolding } = useHoldToCommit(status, label, handleCommit);

    const handleSpace = () => setTextBuffer(prev => prev + ' ');
    const handleBackspace = () => setTextBuffer(prev => prev.slice(0, -1));
    const handleClear = () => setTextBuffer('');

    const handleCopy = async () => {
        if (!textBuffer) return;
        try {
            await navigator.clipboard.writeText(textBuffer);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy text: ', err);
        }
    };

    const handleSpeak = () => {
        if (!textBuffer || !('speechSynthesis' in window)) return;
        
        // This is free and fully client-side. It empowers a signer to "speak" 
        // what they spelled to a hearing person without any internet connection.
        const utterance = new SpeechSynthesisUtterance(textBuffer);
        
        // Try to find an Indian English voice for local context, fallback to any English
        const voices = window.speechSynthesis.getVoices();
        const enIN = voices.find(v => v.lang === 'en-IN');
        if (enIN) {
            utterance.voice = enIN;
        }
        
        window.speechSynthesis.speak(utterance);
    };

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e) => {
            // Only trigger if not typing in an input (though there are no inputs here currently, good practice)
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            if (e.code === 'Space') {
                e.preventDefault();
                handleSpace();
            } else if (e.code === 'Backspace') {
                e.preventDefault();
                handleBackspace();
            } else if (e.code === 'Escape') {
                e.preventDefault();
                handleClear();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    const progressRadius = 18;
    const progressCircumference = 2 * Math.PI * progressRadius;
    const progressOffset = progressCircumference - progress * progressCircumference;

    return (
        <div className="flex flex-col gap-6">
            
            {/* Controls Row */}
            <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                
                {/* Filter Toggle */}
                <div className="flex bg-cream-200 rounded-full p-1 shadow-inner">
                    {['letters', 'numbers', 'both'].map((f) => (
                        <button
                            key={f}
                            onClick={() => setClassFilter(f)}
                            className={`px-4 py-1.5 rounded-full text-sm font-medium capitalize transition-colors duration-200 ${
                                classFilter === f 
                                    ? 'bg-blush-700 text-white shadow-sm' 
                                    : 'text-ink-muted hover:text-ink'
                            }`}
                        >
                            {f}
                        </button>
                    ))}
                </div>

                {/* Status Indicator (Right side) */}
                <div className="flex items-center gap-3">
                    <div className="text-sm text-ink-muted font-medium flex items-center gap-2">
                        {isHolding ? (
                            <span>Holding <strong className="text-ink font-display text-lg">{label}</strong>...</span>
                        ) : (
                            <span>Waiting for sign...</span>
                        )}
                    </div>
                    {/* SVG Progress Ring */}
                    <div className="relative w-10 h-10 flex items-center justify-center">
                        <svg className="w-10 h-10 -rotate-90 transform" viewBox="0 0 44 44">
                            <circle 
                                className="text-blush-100"
                                strokeWidth="4"
                                stroke="currentColor"
                                fill="transparent"
                                r={progressRadius}
                                cx="22"
                                cy="22"
                            />
                            <circle
                                className="text-blush-600 transition-all duration-75 ease-linear"
                                strokeWidth="4"
                                strokeDasharray={progressCircumference}
                                strokeDashoffset={progressOffset}
                                strokeLinecap="round"
                                stroke="currentColor"
                                fill="transparent"
                                r={progressRadius}
                                cx="22"
                                cy="22"
                            />
                        </svg>
                    </div>
                </div>
            </div>

            {/* Text Buffer Display */}
            <div 
                className="bg-cream-50 border border-cream-300 rounded-2xl p-6 md:p-8 min-h-[160px] shadow-[0_4px_24px_rgba(46,34,38,0.05)] relative flex flex-col"
                aria-live="polite"
            >
                <div className="flex-1">
                    {textBuffer ? (
                        <p className="font-display text-4xl md:text-5xl text-ink leading-tight break-all">
                            {textBuffer.slice(0, -1)}
                            <span className={`transition-colors duration-500 ${lastCharCommitted ? 'text-blush-600' : 'text-ink'}`}>
                                {textBuffer.slice(-1)}
                            </span>
                            <span className="inline-block w-[3px] h-10 md:h-12 bg-blush-600 ml-1 translate-y-2 animate-pulse" />
                        </p>
                    ) : (
                        <p className="font-display text-3xl md:text-4xl text-cream-400 leading-tight">
                            Hold a sign to start spelling&hellip;
                            <span className="inline-block w-[3px] h-8 md:h-10 bg-blush-300 ml-1 translate-y-1 animate-pulse" />
                        </p>
                    )}
                </div>
                
                {/* Actions */}
                <div className="flex flex-wrap items-center justify-between gap-4 mt-8 pt-4 border-t border-cream-300/50">
                    <div className="flex gap-2">
                        <button 
                            onClick={handleSpace}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-transparent text-blush-700 border border-cream-300 hover:bg-blush-50 transition-colors text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blush-600"
                            title="Space (Spacebar)"
                        >
                            <Space size={16} /> Space
                        </button>
                        <button 
                            onClick={handleBackspace}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-transparent text-blush-700 border border-cream-300 hover:bg-blush-50 transition-colors text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blush-600"
                            title="Backspace"
                        >
                            <Delete size={16} /> Delete
                        </button>
                        <button 
                            onClick={handleClear}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-transparent text-blush-700 border border-cream-300 hover:bg-blush-50 transition-colors text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blush-600"
                            title="Clear All (Esc)"
                        >
                            <Trash2 size={16} /> Clear
                        </button>
                    </div>
                    
                    <div className="flex gap-3">
                        {canSpeak && (
                            <button 
                                onClick={handleSpeak}
                                disabled={!textBuffer}
                                className="flex items-center gap-2 px-4 py-2 rounded-full bg-cream-50 text-blush-700 border border-blush-200 hover:bg-blush-50 transition-colors font-medium disabled:opacity-50 disabled:hover:bg-cream-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blush-600"
                            >
                                <Volume2 size={18} /> Speak
                            </button>
                        )}
                        <button 
                            onClick={handleCopy}
                            disabled={!textBuffer}
                            className="flex items-center gap-2 px-6 py-2 rounded-full bg-blush-700 text-white hover:bg-blush-800 transition-colors font-medium relative overflow-hidden disabled:opacity-50 disabled:hover:bg-blush-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blush-600"
                        >
                            <Copy size={18} /> {copied ? "Copied!" : "Copy"}
                        </button>
                    </div>
                </div>
            </div>
            
            <div className="flex justify-center gap-6 text-xs text-ink-muted">
                <span><kbd className="font-sans px-1.5 py-0.5 rounded bg-cream-200 text-ink">Space</kbd> Space</span>
                <span><kbd className="font-sans px-1.5 py-0.5 rounded bg-cream-200 text-ink">Backspace</kbd> Delete</span>
                <span><kbd className="font-sans px-1.5 py-0.5 rounded bg-cream-200 text-ink">Esc</kbd> Clear</span>
            </div>

        </div>
    );
}
