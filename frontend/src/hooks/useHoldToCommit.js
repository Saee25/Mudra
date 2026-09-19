import { useState, useEffect, useRef } from 'react';

const HOLD_DURATION_MS = 1500;
const FLICKER_TOLERANCE_MS = 250;
const COOLDOWN_MS = 1000;

export function useHoldToCommit(status, label, onCommit) {
    const [progress, setProgress] = useState(0);
    const holdTimerRef = useRef(null);
    const flickerTimerRef = useRef(null);
    const cooldownTimerRef = useRef(null);
    const lastCommittedRef = useRef(null);
    const startTimeRef = useRef(null);
    const holdingLabelRef = useRef(null);
    const rafRef = useRef(null);

    // Clean up function to clear all timers and state
    const resetHold = () => {
        if (holdTimerRef.current) clearTimeout(holdTimerRef.current);
        if (flickerTimerRef.current) clearTimeout(flickerTimerRef.current);
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        
        holdTimerRef.current = null;
        flickerTimerRef.current = null;
        startTimeRef.current = null;
        holdingLabelRef.current = null;
        setProgress(0);
    };

    // Update progress smoothly for the circular indicator
    const updateProgress = () => {
        if (!startTimeRef.current) return;
        const elapsed = performance.now() - startTimeRef.current;
        const currentProgress = Math.min(elapsed / HOLD_DURATION_MS, 1);
        setProgress(currentProgress);
        
        if (currentProgress < 1) {
            rafRef.current = requestAnimationFrame(updateProgress);
        }
    };

    useEffect(() => {
        // Condition 1: Lost confidence or hand disappeared.
        // We use a flicker tolerance. If a hand briefly drops out (which happens
        // often in 2-handed signs) we don't immediately cancel the hold progress.
        if (status !== 'confident' || !label) {
            if (holdingLabelRef.current && !flickerTimerRef.current) {
                // Start a flicker tolerance timer
                flickerTimerRef.current = setTimeout(() => {
                    resetHold();
                }, FLICKER_TOLERANCE_MS);
            }
            return;
        }

        // We have a confident label!
        
        // If the label changed to something new during a flicker, reset immediately
        if (holdingLabelRef.current && holdingLabelRef.current !== label) {
            resetHold();
        }

        // If we recently committed this label, we need a cooldown or a label change 
        // to prevent holding "L" from typing "LLLLLLL" rapidly. 
        // Users can type double letters ("APPLE") by dropping their hands or waiting 
        // out the cooldown.
        if (lastCommittedRef.current === label && cooldownTimerRef.current) {
            return; // Still in cooldown for this specific label, do not start hold.
        }
        
        // Clear cooldown if the label changed
        if (lastCommittedRef.current !== label) {
            if (cooldownTimerRef.current) {
                clearTimeout(cooldownTimerRef.current);
                cooldownTimerRef.current = null;
            }
            lastCommittedRef.current = null;
        }

        // If a flicker timer is running but the hand came back with the SAME label,
        // cancel the flicker timer and resume holding!
        if (flickerTimerRef.current && holdingLabelRef.current === label) {
            clearTimeout(flickerTimerRef.current);
            flickerTimerRef.current = null;
            return; // Continue existing hold
        }

        // Start a new hold
        if (!holdingLabelRef.current) {
            holdingLabelRef.current = label;
            startTimeRef.current = performance.now();
            
            rafRef.current = requestAnimationFrame(updateProgress);

            holdTimerRef.current = setTimeout(() => {
                // Success! Hold completed.
                onCommit(label);
                
                // Set cooldown for this specific label
                lastCommittedRef.current = label;
                cooldownTimerRef.current = setTimeout(() => {
                    cooldownTimerRef.current = null;
                    lastCommittedRef.current = null; // Clear so they can hold again if they just kept holding
                }, COOLDOWN_MS);
                
                resetHold();
            }, HOLD_DURATION_MS);
        }

    }, [status, label, onCommit]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            resetHold();
            if (cooldownTimerRef.current) clearTimeout(cooldownTimerRef.current);
        };
    }, []);

    return { progress, isHolding: progress > 0 };
}
