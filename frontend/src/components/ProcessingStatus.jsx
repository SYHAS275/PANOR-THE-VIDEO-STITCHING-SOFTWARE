import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Loader2, Sparkles } from 'lucide-react';

export default function ProcessingStatus({ jobId, apiBase, onComplete, onError }) {
    const [progress, setProgress] = useState(0);
    const [message, setMessage] = useState('Initializing...');
    const [logs, setLogs] = useState([]);

    useEffect(() => {
        if (!jobId) return;

        const interval = setInterval(async () => {
            try {
                const res = await axios.get(`${apiBase}/status/${jobId}`);
                const data = res.data;

                setProgress(data.progress);
                setMessage(data.message);

                // Add unique messages to logs
                if (data.message && logs[logs.length - 1] !== data.message) {
                    setLogs(prev => [...prev.slice(-4), data.message]);
                }

                if (data.status === 'completed') {
                    clearInterval(interval);
                    onComplete(data.output_file);
                } else if (data.status === 'failed') {
                    clearInterval(interval);
                    onError(data.error || 'Processing failed');
                }
            } catch (err) {
                console.error("Polling error", err);
            }
        }, 1000);

        return () => clearInterval(interval);
    }, [jobId, apiBase, onComplete, onError, logs]);

    return (
        <div className="w-full max-w-lg mx-auto space-y-8 animate-fade-in glass-panel p-10 rounded-lg relative overflow-hidden">
            {/* Ambient neon glow */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-[#00D4FF]/10 rounded-full blur-[60px] pointer-events-none" />

            <div className="text-center space-y-4 relative z-10">
                <div className="relative inline-flex justify-center items-center">
                    <div className="absolute inset-0 bg-[#00D4FF]/20 blur-2xl rounded-full animate-neon-pulse"></div>
                    <div className="relative bg-black p-4 rounded-lg border border-[#222222] shadow-[0_0_20px_rgba(0,212,255,0.2)]">
                        <Loader2 className="w-12 h-12 text-[#00D4FF] animate-spin" strokeWidth={1.5} />
                    </div>
                </div>
                <div>
                     <h2 className="text-3xl font-bold text-gradient-white tracking-tight">Stitching in Progress</h2>
                    <p className="text-gradient-subtle mt-2 font-light">Alignment & Blending engine running...</p>
                </div>
            </div>

            {/* Progress bar with neon effect */}
            <div className="space-y-3 relative z-10">
                <div className="flex justify-between text-sm font-bold tracking-widest text-white/50 uppercase">
                    <span className="flex items-center gap-2">
                        <Sparkles className="w-3 h-3 text-[#00D4FF]" strokeWidth={1.5} />
                        Processing
                    </span>
                    <span className="text-[#00D4FF]">{progress}%</span>
                </div>
                <div className="h-2 bg-black rounded-full overflow-hidden border border-[#222222]">
                    <div
                        className="h-full progress-neon transition-all duration-500 ease-out relative"
                        style={{ width: `${progress}%` }}
                    >
                        <div className="absolute top-0 right-0 bottom-0 w-[40px] bg-gradient-to-r from-transparent to-white/60 blur-[2px]" />
                    </div>
                </div>
            </div>

            {/* Terminal-style log output */}
            <div className="bg-black border border-[#222222] rounded-lg p-5 font-mono text-xs space-y-2 h-40 overflow-hidden flex flex-col justify-end relative z-10">
                {logs.map((log, i) => (
                    <div key={i} className="text-white/30 truncate animate-slide-up border-l-2 border-[#222222] pl-3">
                        <span className="text-[#00D4FF]/50 mr-2">$</span>{log}
                    </div>
                ))}
                <div className="text-white font-semibold truncate animate-pulse border-l-2 border-[#00D4FF] pl-3">
                    <span className="text-[#00D4FF] mr-2">&gt;</span>{message}
                </div>
            </div>
        </div>
    );
}
