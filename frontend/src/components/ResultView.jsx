import React from 'react';
import { Download, RotateCcw, Check, Play, Share2 } from 'lucide-react';

export default function ResultView({ filename, apiBase, onReset }) {
    const downloadUrl = `${apiBase}/download/${filename}`;

    return (
        <div className="space-y-8 animate-scale-in w-full max-w-5xl mx-auto">
            <div className="text-center space-y-4">
                <div className="inline-flex p-4 bg-[#39FF14]/10 text-[#39FF14] rounded-lg mb-2 border border-[#39FF14]/30 shadow-[0_0_30px_rgba(57,255,20,0.2)] animate-neon-pulse">
                    <Check className="w-10 h-10" strokeWidth={1.5} />
                </div>
                <h2 className="text-5xl font-bold text-gradient-lime tracking-tight">Stitching Complete</h2>
                <p className="text-gradient-subtle text-xl font-light">Your cinematic panorama is ready.</p>
            </div>

            {/* Video preview with cinematic border */}
            <div className="relative group rounded-lg overflow-hidden border border-[#222222] bg-black shadow-[0_0_40px_rgba(0,0,0,0.8)] aspect-video">
                 {/* Glow effect behind video */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent pointer-events-none z-10" />
                
                <video
                    src={downloadUrl}
                    controls
                    className="w-full h-full object-contain relative z-0"
                    poster="" 
                />

                {/* Corner accents for cinematic feel */}
                <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-[#00D4FF]/50 pointer-events-none z-20"></div>
                <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-[#00D4FF]/50 pointer-events-none z-20"></div>
                <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-[#00D4FF]/50 pointer-events-none z-20"></div>
                <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-[#00D4FF]/50 pointer-events-none z-20"></div>
            </div>

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
