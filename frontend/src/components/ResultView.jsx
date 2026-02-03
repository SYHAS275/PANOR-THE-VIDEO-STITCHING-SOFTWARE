import React, { useEffect, useState, useRef } from 'react';
import { Download, RotateCcw, Check, Play, Pause } from 'lucide-react';

export default function ResultView({ filename, apiBase, jobId, onReset }) {
    const downloadUrl = `${apiBase}/download/${filename}`;
    const [sources, setSources] = useState([]);
    const mainRef = useRef(null);
    const sourceRefs = useRef([]);

    const handlePlayAll = () => {
        const videos = [mainRef.current, ...sourceRefs.current.filter(Boolean)];
        videos.forEach(v => {
            try {
                v.currentTime = 0;
                const p = v.play();
                if (p && p.catch) p.catch(() => {});
            } catch {}
        });
    };

    const handlePauseAll = () => {
        const videos = [mainRef.current, ...sourceRefs.current.filter(Boolean)];
        videos.forEach(v => {
            try {
                v.pause();
            } catch {}
        });
    };
    useEffect(() => {
        let active = true;
        const run = async () => {
            if (!jobId) return;
            try {
                const res = await fetch(`${apiBase}/sources/${jobId}`);
                const data = await res.json();
                if (!active) return;
                const list = (data?.sources || []).map(s => ({
                    ...s,
                    url: `${apiBase}/source/${jobId}/${s.index}?preview=true`
                }));
                setSources(list);
            } catch (_) {}
        };
        run();
        return () => { active = false; };
    }, [jobId, apiBase]);

    const colsClass = sources.length === 3 ? 'grid-cols-1 sm:grid-cols-3' : 'grid-cols-1 sm:grid-cols-2';

    return (
        <div className="space-y-8 animate-scale-in w-full max-w-5xl mx-auto">
            <div className="text-center space-y-4">
                <div className="inline-flex p-4 bg-[#39FF14]/10 text-[#39FF14] rounded-lg mb-2 border border-[#39FF14]/30 shadow-[0_0_30px_rgba(57,255,20,0.2)] animate-neon-pulse">
                    <Check className="w-10 h-10" strokeWidth={1.5} />
                </div>
                <h2 className="text-5xl font-bold text-gradient-lime tracking-tight leading-tight pb-2">Stitching Complete</h2>
                <p className="text-gradient-subtle text-xl font-light">Your cinematic panorama is ready.</p>
            </div>

            <div className="flex justify-center gap-4">
                <button
                    onClick={handlePlayAll}
                    className="px-4 py-2 rounded-lg bg-[#00D4FF] text-black font-semibold text-sm hover:bg-[#00D4FF]/90 transition-all flex items-center gap-2"
                >
                    <Play className="w-4 h-4" />
                    Play All
                </button>
                <button
                    onClick={handlePauseAll}
                    className="px-4 py-2 rounded-lg bg-red-500 text-white font-semibold text-sm hover:bg-red-500/90 transition-all flex items-center gap-2"
                >
                    <Pause className="w-4 h-4" />
                    Pause All
                </button>
            </div>

            {/* Video preview with cinematic border */}
            <div className="relative group rounded-lg overflow-hidden border border-[#222222] bg-black shadow-[0_0_40px_rgba(0,0,0,0.8)] aspect-video">
                 {/* Glow effect behind video */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent pointer-events-none z-10" />
                
                <video
                    ref={mainRef}
                    src={downloadUrl}
                    controls
                    autoPlay
                    muted
                    loop
                    playsInline
                    crossOrigin="anonymous"
                    className="w-full h-full object-contain relative z-0"
                    poster=""
                />

                {/* Corner accents for cinematic feel */}
                <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-[#00D4FF]/50 pointer-events-none z-20"></div>
                <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-[#00D4FF]/50 pointer-events-none z-20"></div>
                <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-[#00D4FF]/50 pointer-events-none z-20"></div>
                <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-[#00D4FF]/50 pointer-events-none z-20"></div>
            </div>

            {sources.length > 0 && (
                <div className={`grid ${colsClass} gap-6`}>
                    {sources.map(src => (
                        <div key={src.index} className="relative group rounded-lg overflow-hidden border border-[#222222] bg-black shadow-[0_0_30px_rgba(0,0,0,0.6)]">
                            <div className="px-4 py-2 border-b border-[#222222] text-sm text-white/70">
                                Camera {src.index === 0 ? 'Left' : src.index === 1 && sources.length === 3 ? 'Center' : 'Right'}
                            </div>
                            <video
                                ref={el => { sourceRefs.current[src.index] = el; }}
                                src={src.url}
                                controls
                                autoPlay
                                muted
                                loop
                                playsInline
                                crossOrigin="anonymous"
                                className="w-full h-full object-contain"
                            />
                        </div>
                    ))}
                </div>
            )}

            {/* Action buttons */}
            <div className="flex flex-col sm:flex-row gap-6 justify-center pt-4">
                <a
                    href={`${downloadUrl}?download=true`}
                    download
                    className="px-8 py-4 btn-neon-lime rounded-lg font-bold flex items-center justify-center gap-3 hover:scale-[1.02] transition-all group"
                >
                    <Download className="w-5 h-5 group-hover:animate-bounce" strokeWidth={1.5} />
                    Download Video
                </a>
                
                <button
                    onClick={onReset}
                    className="px-8 py-4 btn-cinema-outline rounded-lg font-semibold flex items-center justify-center gap-3 transition-all"
                >
                    <RotateCcw className="w-5 h-5" strokeWidth={1.5} />
                    Stitch Another
                </button>
            </div>
        </div>
    );
}
