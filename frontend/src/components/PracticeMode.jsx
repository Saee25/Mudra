import React, { useState, useEffect, useCallback } from 'react';
import { getLabels, isSpecialClass, getClassIndices, isLetter, isDigit } from '../utils/labels';
import { useHoldToCommit } from '../hooks/useHoldToCommit';
import { SIGN_HINTS, MOTION_SIGNS } from '../data/signHints';
import { CheckCircle2, ChevronRight, HelpCircle } from 'lucide-react';

export default function PracticeMode({ status, label, classFilter, setClassFilter }) {
    const [targetSign, setTargetSign] = useState(null);
    const [score, setScore] = useState(0);
    const [streak, setStreak] = useState(0);
    const [bestStreak, setBestStreak] = useState(0);
    
    const [successState, setSuccessState] = useState(false);
    const [showHint, setShowHint] = useState(false);

    // Pick a random target from the allowed set
    const pickTarget = useCallback((currentFilter, currentTarget = null) => {
        const labels = getLabels();
        if (!labels || labels.length === 0) return null;

        const indices = getClassIndices(currentFilter);
        const candidates = indices
            .map(i => labels[i])
            .filter(l => !isSpecialClass(l) && !MOTION_SIGNS.includes(l) && l !== currentTarget);
        
        if (candidates.length === 0) return null;
        return candidates[Math.floor(Math.random() * candidates.length)];
    }, []);

    // Initialize target when labels load or filter changes
    useEffect(() => {
        setTargetSign(pickTarget(classFilter, targetSign));
        setSuccessState(false);
        setShowHint(false);
        // We do not reset the score/streak on filter change, to let users seamlessly switch
    }, [classFilter, pickTarget]);

    const handleSuccess = useCallback((committedLabel) => {
        // Only trigger if they matched the target
        if (committedLabel !== targetSign) return;

        setSuccessState(true);
        setScore(s => s + 1);
        setStreak(s => {
            const newStreak = s + 1;
            setBestStreak(b => Math.max(b, newStreak));
            return newStreak;
        });

        // Small pause before next target
        setTimeout(() => {
            setTargetSign(pickTarget(classFilter, targetSign));
            setSuccessState(false);
            setShowHint(false);
        }, 1200);
    }, [targetSign, classFilter, pickTarget]);

    const handleSkip = () => {
        setTargetSign(pickTarget(classFilter, targetSign));
        setStreak(0);
        setSuccessState(false);
        setShowHint(false);
    };

    // We only pass handleSuccess to useHoldToCommit. It will only fire if the user 
    // holds *any* label, but we filter inside handleSuccess to check if it's the target.
    // However, to only show progress when they hold the *correct* sign, we pass 
    // a pseudo-status.
    const isTargetHeld = status === 'confident' && label === targetSign;
    const { progress, isHolding } = useHoldToCommit(
        isTargetHeld ? 'confident' : 'uncertain', 
        isTargetHeld ? label : '', 
        handleSuccess
    );

    const progressRadius = 18;
    const progressCircumference = 2 * Math.PI * progressRadius;
    const progressOffset = progressCircumference - progress * progressCircumference;

    return (
        <div className="flex flex-col gap-6">
            
            {/* Controls Row */}
            <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
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

                <div className="flex gap-6 text-center">
                    <div className="flex flex-col">
                        <span className="font-display text-2xl text-ink leading-none">{score}</span>
                        <span className="text-[10px] uppercase tracking-wider text-ink-muted font-medium mt-1">Score</span>
                    </div>
                    <div className="flex flex-col">
                        <span className="font-display text-2xl text-ink leading-none">{streak}</span>
                        <span className="text-[10px] uppercase tracking-wider text-ink-muted font-medium mt-1">Streak</span>
                    </div>
                    <div className="flex flex-col">
                        <span className="font-display text-2xl text-ink leading-none">{bestStreak}</span>
                        <span className="text-[10px] uppercase tracking-wider text-ink-muted font-medium mt-1">Best</span>
                    </div>
                </div>
            </div>

            {/* Practice Card */}
            <div className={`border rounded-2xl p-6 md:p-8 min-h-[220px] shadow-[0_4px_24px_rgba(46,34,38,0.05)] relative flex flex-col items-center justify-center transition-colors duration-300 ${
                successState ? 'bg-cream-50 border-sage/40' : 'bg-cream-50 border-cream-300'
            }`}>
                
                {successState ? (
                    <div className="flex flex-col items-center justify-center transition-all duration-300 scale-100 opacity-100">
                        <CheckCircle2 size={64} className="text-sage mb-4" />
                        <h3 className="font-display text-3xl text-sage">Nice!</h3>
                    </div>
                ) : (
                    <>
                        <span className="text-xs uppercase tracking-widest text-ink-muted font-semibold mb-2">Show me</span>
                        <h2 className="font-display text-7xl md:text-8xl text-blush-800 leading-none mb-6">
                            {targetSign || '-'}
                        </h2>

                        {/* Hint Section */}
                        {showHint ? (
                            <div className="mt-2 text-center max-w-sm">
                                <p className="text-ink font-medium text-sm mb-1">{SIGN_HINTS[targetSign]}</p>
                            </div>
                        ) : (
                            <button 
                                onClick={() => setShowHint(true)}
                                className="flex items-center gap-1.5 text-xs font-medium text-ink-muted hover:text-blush-700 transition-colors"
                            >
                                <HelpCircle size={14} /> Show hint
                            </button>
                        )}
                        
                        {/* Progress SVG */}
                        <div className="absolute top-6 right-6 flex items-center justify-center w-12 h-12">
                            <svg className="w-12 h-12 -rotate-90 transform" viewBox="0 0 44 44">
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
                    </>
                )}

            </div>

            {/* Actions */}
            <div className="flex items-center justify-between">
                <p className="text-xs text-ink-muted max-w-[200px] leading-tight">
                    For accurate hand shapes, please refer to the official <a href="https://islrtc.nic.in/" target="_blank" rel="noreferrer" className="underline hover:text-ink">ISLRTC resources</a>.
                </p>
                <button 
                    onClick={handleSkip}
                    disabled={successState}
                    className="flex items-center gap-1 px-4 py-2 rounded-full bg-transparent text-ink-muted border border-cream-300 hover:bg-cream-200 hover:text-ink transition-colors font-medium text-sm disabled:opacity-50"
                >
                    Skip <ChevronRight size={16} />
                </button>
            </div>

        </div>
    );
}
