import { useState, useEffect, useRef } from 'react';

const HOLD_DURATION_MS = 1500;
const FLICKER_TOLERANCE_MS = 250;
const COOLDOWN_MS = 1000;

export function useHoldToCommit(status, label, onCommit) {
    const [progress, setProgress] = useState(0);
    
    // Store latest props in refs so the RAF loop always sees the freshest values
    const latestStatus = useRef(status);
    const latestLabel = useRef(label);
    const latestOnCommit = useRef(onCommit);
    
    useEffect(() => {
        latestStatus.current = status;
        latestLabel.current = label;
        latestOnCommit.current = onCommit;
    }, [status, label, onCommit]);

    useEffect(() => {
        let rafId;
        
        let holdingLabel = null;
        let holdStartTime = 0;
        let flickerStartTime = 0;
        let lastCommittedLabel = null;
        let cooldownEndTime = 0;
        
        const loop = () => {
            const now = performance.now();
            const currentStatus = latestStatus.current;
            const currentLabel = latestLabel.current;
            
            // 1. Check cooldown
            if (lastCommittedLabel && now < cooldownEndTime) {
                // If they change signs during cooldown, clear cooldown
                if (currentLabel && currentLabel !== lastCommittedLabel) {
                    lastCommittedLabel = null;
                } else {
                    // Still in cooldown for this sign
                    setProgress(0);
                    rafId = requestAnimationFrame(loop);
                    return;
                }
            }
            
            // 2. Are we currently seeing a valid sign?
            const isValid = currentStatus === 'confident' && currentLabel;
            
            if (holdingLabel) {
                // We are currently holding a sign
                if (isValid && currentLabel === holdingLabel) {
                    // Holding steadily! Clear any flicker timer.
                    flickerStartTime = 0;
                    
                    // Update progress
                    const elapsed = now - holdStartTime;
                    const p = Math.min(elapsed / HOLD_DURATION_MS, 1);
                    setProgress(p);
                    
                    if (p >= 1) {
                        // COMMIT!
                        latestOnCommit.current(holdingLabel);
                        lastCommittedLabel = holdingLabel;
                        cooldownEndTime = now + COOLDOWN_MS;
                        
                        // Reset hold
                        holdingLabel = null;
                        setProgress(0);
                    }
                } else {
                    // The sign disappeared or changed! Start/check flicker tolerance.
                    if (!flickerStartTime) {
                        flickerStartTime = now;
                    } else if (now - flickerStartTime > FLICKER_TOLERANCE_MS) {
                        // Flicker tolerance exceeded, cancel hold
                        holdingLabel = null;
                        flickerStartTime = 0;
                        setProgress(0);
                    }
                    // While flickering, we don't update progress, it stays frozen
                }
            } else {
                // We are NOT holding a sign. Should we start?
                if (isValid) {
                    holdingLabel = currentLabel;
                    holdStartTime = now;
                    flickerStartTime = 0;
                    setProgress(0);
                } else {
                    setProgress(0);
                }
            }
            
            rafId = requestAnimationFrame(loop);
        };
        
        rafId = requestAnimationFrame(loop);
        
        return () => cancelAnimationFrame(rafId);
    }, []); // Run once!

    return { progress, isHolding: progress > 0 };
}
