import React, { useState, useEffect, useRef } from 'react';
import { getLabels } from '../utils/labels';
import { cropHands } from '../utils/handCrop';

export default function DataCollector({ videoRef, lastInput, cropBox }) {
    const [signerId, setSignerId] = useState('s1');
    const [selectedLabel, setSelectedLabel] = useState('A');
    const [samplesPerBurst, setSamplesPerBurst] = useState(100);
    const [captureRate, setCaptureRate] = useState(5);
    const [saveCrops, setSaveCrops] = useState(false);
    
    const [isRecording, setIsRecording] = useState(false);
    const [countdown, setCountdown] = useState(0);
    const [sessionSamples, setSessionSamples] = useState([]);
    const [burstCount, setBurstCount] = useState(0);
    
    // We want a stable session ID
    const sessionIdRef = useRef(`ses_${Date.now()}`);
    const recordingInterval = useRef(null);
    const lastInputRef = useRef(lastInput);
    const cropBoxRef = useRef(cropBox);
    const videoRefCurrent = useRef(videoRef?.current);

    useEffect(() => {
        lastInputRef.current = lastInput;
        cropBoxRef.current = cropBox;
        videoRefCurrent.current = videoRef?.current;
    }, [lastInput, cropBox, videoRef]);

    useEffect(() => {
        const handleBeforeUnload = (e) => {
            if (sessionSamples.length > 0) {
                e.preventDefault();
                e.returnValue = '';
            }
        };
        window.addEventListener('beforeunload', handleBeforeUnload);
        return () => window.removeEventListener('beforeunload', handleBeforeUnload);
    }, [sessionSamples]);

    // Handle Countdown -> Recording
    useEffect(() => {
        if (countdown > 0) {
            const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
            return () => clearTimeout(timer);
        } else if (countdown === 0 && document.getElementById('collect-btn')?.dataset.pending === 'true') {
            document.getElementById('collect-btn').dataset.pending = 'false';
            setIsRecording(true);
            setBurstCount(0);
        }
    }, [countdown]);

    // Handle Recording Loop
    useEffect(() => {
        if (!isRecording) return;
        const intervalMs = 1000 / captureRate;
        recordingInterval.current = setInterval(() => {
            setBurstCount(c => {
                if (c >= samplesPerBurst - 1) {
                    stopRecording();
                    captureSample(); // Capture the last one
                    return c + 1;
                }
                captureSample();
                return c + 1;
            });
        }, intervalMs);
        return () => clearInterval(recordingInterval.current);
    }, [isRecording, captureRate, samplesPerBurst]);

    const startRecording = () => {
        if (!signerId.trim()) {
            alert('Please enter a signer ID (e.g. s1, s2)');
            return;
        }
        document.getElementById('collect-btn').dataset.pending = 'true';
        setCountdown(3);
    };

    const stopRecording = () => {
        setIsRecording(false);
        if (recordingInterval.current) clearInterval(recordingInterval.current);
    };

    const captureSample = () => {
        const input = lastInputRef.current;
        const box = cropBoxRef.current;
        const video = videoRefCurrent.current;

        if (!input || !input.raw || input.built.slots === 0) return; // Need at least 1 hand

        let crop_jpeg = undefined;
        if (saveCrops && box && video) {
            try {
                const canvas = cropHands(video, box);
                crop_jpeg = canvas.toDataURL('image/jpeg', 0.95);
            } catch (e) {
                console.warn("Crop failed", e);
            }
        }

        // Reshape Float32Array to nested arrays [2][21][3]
        const lmsFlat = input.built.landmarks;
        const landmarks = [];
        for (let i = 0; i < 2; i++) {
            const hand = [];
            for (let j = 0; j < 21; j++) {
                const idx = i * 21 * 3 + j * 3;
                hand.push([lmsFlat[idx], lmsFlat[idx+1], lmsFlat[idx+2]]);
            }
            landmarks.push(hand);
        }

        const sample = {
            label: selectedLabel,
            t_ms: Date.now(),
            raw_landmarks: JSON.parse(JSON.stringify(input.raw)), // Clone
            handedness: JSON.parse(JSON.stringify(input.handedness || [])), // Clone
            landmarks: landmarks,
            hand_mask: Array.from(input.built.handMask),
            crop_jpeg
        };

        setSessionSamples(prev => [...prev, sample]);
    };

    const downloadJSON = () => {
        if (sessionSamples.length === 0) return;
        
        const video = videoRefCurrent.current;
        const data = {
            schema_version: 1,
            pipeline_version: "landmark-v1",
            app: "Mudra",
            session_id: sessionIdRef.current,
            signer_id: signerId,
            created_at: new Date().toISOString(),
            frame_w: video ? video.videoWidth : 0,
            frame_h: video ? video.videoHeight : 0,
            detector_settings: {
                num_hands: 2,
                min_hand_detection_confidence: 0.3,
                min_hand_presence_confidence: 0.3
            },
            samples: sessionSamples
        };

        const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const dateStr = new Date().toISOString().split('T')[0];
        a.href = url;
        a.download = `mudra_collect_${signerId}_${sessionIdRef.current}_${dateStr}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const clearSession = () => {
        if (confirm("Are you sure you want to clear all recorded samples in this session?")) {
            setSessionSamples([]);
        }
    };

    const labels = [...getLabels(), 'other'];
    const counts = {};
    sessionSamples.forEach(s => {
        counts[s.label] = (counts[s.label] || 0) + 1;
    });

    return (
        <div className="bg-cream-50 rounded-2xl p-6 shadow-sm border border-cream-300 w-full flex flex-col gap-6">
            <div className="flex flex-col gap-2">
                <h2 className="text-2xl font-display font-semibold text-blush-800">Collect Data</h2>
                <div className="bg-blush-50 border border-blush-200 text-blush-800 p-3 rounded-lg text-sm">
                    <strong>Privacy Note:</strong> This tool records hand landmark coordinates (and optional small crops) directly in your browser. Nothing is uploaded to any server. Please obtain consent if recording other people.
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex flex-col gap-1">
                    <label className="text-sm font-medium text-ink-muted">Signer ID (e.g. s1, s2)</label>
                    <input 
                        type="text" 
                        value={signerId} 
                        onChange={(e) => setSignerId(e.target.value)} 
                        className="p-2 border border-cream-300 rounded-lg bg-white outline-none focus:border-blush-400"
                        placeholder="s1"
                    />
                </div>
                <div className="flex flex-col gap-1">
                    <label className="text-sm font-medium text-ink-muted">Label to Record</label>
                    <select 
                        value={selectedLabel} 
                        onChange={(e) => setSelectedLabel(e.target.value)}
                        className="p-2 border border-cream-300 rounded-lg bg-white outline-none focus:border-blush-400"
                    >
                        {labels.map(l => <option key={l} value={l}>{l}</option>)}
                    </select>
                </div>
                <div className="flex flex-col gap-1">
                    <label className="text-sm font-medium text-ink-muted">Samples per burst</label>
                    <input 
                        type="number" 
                        value={samplesPerBurst} 
                        onChange={(e) => setSamplesPerBurst(Number(e.target.value))} 
                        className="p-2 border border-cream-300 rounded-lg bg-white outline-none focus:border-blush-400"
                    />
                </div>
                <div className="flex flex-col gap-1">
                    <label className="text-sm font-medium text-ink-muted">Capture rate (FPS)</label>
                    <input 
                        type="number" 
                        value={captureRate} 
                        onChange={(e) => setCaptureRate(Number(e.target.value))} 
                        className="p-2 border border-cream-300 rounded-lg bg-white outline-none focus:border-blush-400"
                    />
                    <span className="text-xs text-ink-muted">Keep ~5 FPS to avoid recording duplicate frames.</span>
                </div>
                <div className="flex items-center gap-2 md:col-span-2 mt-2">
                    <input 
                        type="checkbox" 
                        id="saveCrops" 
                        checked={saveCrops} 
                        onChange={(e) => setSaveCrops(e.target.checked)} 
                        className="w-4 h-4 accent-blush-600"
                    />
                    <label htmlFor="saveCrops" className="text-sm font-medium text-ink">Also save 224x224 hand crops (creates large files, only needed for MobileNetV2 eval)</label>
                </div>
            </div>

            <div className="flex items-center gap-4 py-4 border-y border-cream-200">
                <button 
                    id="collect-btn"
                    onClick={startRecording} 
                    disabled={isRecording || countdown > 0}
                    className="px-6 py-2 bg-blush-700 hover:bg-blush-800 text-white rounded-full font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {countdown > 0 ? `Starting in ${countdown}...` : isRecording ? 'Recording...' : 'Record'}
                </button>
                <button 
                    onClick={stopRecording} 
                    disabled={!isRecording && countdown === 0}
                    className="px-6 py-2 bg-cream-50 border border-cream-300 text-ink rounded-full font-medium hover:bg-cream-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    Stop
                </button>
                {isRecording && (
                    <div className="flex-1 flex items-center gap-2">
                        <div className="w-full bg-cream-200 h-2 rounded-full overflow-hidden">
                            <div 
                                className="bg-blush-500 h-full transition-all" 
                                style={{ width: `${(burstCount / samplesPerBurst) * 100}%` }}
                            ></div>
                        </div>
                        <span className="text-sm font-medium text-ink-muted min-w-[3rem] text-right">{burstCount}/{samplesPerBurst}</span>
                    </div>
                )}
            </div>

            <div className="flex flex-col gap-2">
                <h3 className="text-lg font-display font-medium text-ink">Session Progress</h3>
                <p className="text-xs text-ink-muted">
                    Suggestion: Record 2-3 bursts per label, moving your hand slightly and changing distance/lighting between bursts.
                </p>
                <div className="flex flex-wrap gap-2 mt-2">
                    {labels.map(l => (
                        <div key={l} className={`px-3 py-1 rounded-full text-sm font-medium border ${counts[l] ? 'bg-blush-100 text-blush-800 border-blush-200' : 'bg-cream-100 text-ink-muted border-cream-200'}`}>
                            {l}: {counts[l] || 0}
                        </div>
                    ))}
                </div>
                <div className="mt-2 text-sm font-medium text-ink">
                    Total Session Samples: {sessionSamples.length}
                </div>
            </div>

            <div className="flex items-center gap-4 mt-2">
                <button 
                    onClick={downloadJSON} 
                    disabled={sessionSamples.length === 0}
                    className="px-6 py-2 bg-blush-700 hover:bg-blush-800 text-white rounded-full font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    Download JSON
                </button>
                <button 
                    onClick={clearSession} 
                    disabled={sessionSamples.length === 0}
                    className="px-6 py-2 bg-transparent text-blush-700 font-medium transition-colors hover:underline disabled:opacity-50 disabled:cursor-not-allowed disabled:no-underline"
                >
                    Clear session
                </button>
            </div>
        </div>
    );
}
