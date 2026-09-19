import React, { useEffect, useRef, useState, forwardRef } from 'react';

const Webcam = forwardRef(({ onReady, className = '' }, forwardedRef) => {
    const videoRef = useRef(null);
    const [status, setStatus] = useState('loading'); // 'loading', 'ready', 'error'
    const [errorMessage, setErrorMessage] = useState('');

    useEffect(() => {
        let stream = null;

        const startCamera = async () => {
            try {
                setStatus('loading');
                
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
                        throw new Error('insecure');
                    }
                    throw new Error('unsupported');
                }

                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    },
                    audio: false
                });

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            } catch (err) {
                console.error("Webcam error:", err);
                setStatus('error');
                if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                    setErrorMessage('Camera access was denied. Please allow camera access in your browser settings to use Mudra.');
                } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                    setErrorMessage('No camera found. Please connect a camera to your device.');
                } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
                    setErrorMessage('Your camera is in use by another application. Please close it and try again.');
                } else if (err.message === 'insecure') {
                    setErrorMessage('Camera access requires a secure connection (HTTPS or localhost).');
                } else {
                    setErrorMessage('Could not start the camera. Please try again.');
                }
            }
        };

        startCamera();

        return () => {
            // Cleanup on unmount
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    // Expose the video element to parent
    useEffect(() => {
        if (forwardedRef) {
            if (typeof forwardedRef === 'function') {
                forwardedRef(videoRef.current);
            } else {
                forwardedRef.current = videoRef.current;
            }
        }
    }, [forwardedRef]);

    const handleLoadedData = () => {
        setStatus('ready');
        if (onReady) {
            onReady();
        }
    };

    const retryCamera = () => {
        setStatus('loading');
        // A simple reload is often the most robust way to re-request permissions
        // or re-acquire the camera if it was busy.
        window.location.reload(); 
    };

    return (
        <div className={`relative overflow-hidden rounded-2xl border border-cream-300 shadow-[0_4px_24px_rgba(46,34,38,0.05)] bg-cream-50 ${className}`}>
            
            {/* 
              * VIDEO ELEMENT
              * 
              * We use CSS scale-x-[-1] (Tailwind: -scale-x-100) to mirror the display for comfort.
              * IMPORTANT: This only mirrors the *display*. The underlying pixels from the video element 
              * given to MediaPipe are the ORIGINAL, unmirrored frames. This perfectly matches the 
              * training data distribution (where MIRROR_AUGMENT handles left/right hands).
              */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                onLoadedData={handleLoadedData}
                className={`w-full h-full object-cover -scale-x-100 transition-opacity duration-300 ${status === 'ready' ? 'opacity-100' : 'opacity-0'}`}
            />

            {/* Loading State */}
            {status === 'loading' && (
                <div className="absolute inset-0 flex items-center justify-center bg-cream-100 animate-pulse">
                    <p className="text-ink-muted font-medium text-lg font-sans">Starting camera&hellip;</p>
                </div>
            )}

            {/* Error State */}
            {status === 'error' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-cream-50 z-10">
                    <div className="w-16 h-16 mb-4 rounded-full bg-blush-100 flex items-center justify-center">
                        {/* Simple offline SVG icon in blush-700 */}
                        <svg className="w-8 h-8 text-blush-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4l16 16" />
                        </svg>
                    </div>
                    <h3 className="text-xl font-display text-blush-800 mb-2">Camera Unavailable</h3>
                    <p className="text-ink-muted font-sans mb-6 max-w-sm">{errorMessage}</p>
                    <button 
                        onClick={retryCamera}
                        className="px-6 py-2 rounded-full bg-blush-700 text-white font-medium hover:bg-blush-800 transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blush-600 focus:ring-offset-cream-50"
                    >
                        Try again
                    </button>
                </div>
            )}
        </div>
    );
});

Webcam.displayName = 'Webcam';

export default Webcam;
