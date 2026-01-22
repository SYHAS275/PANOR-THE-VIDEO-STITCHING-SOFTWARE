import React, { useState } from 'react';
import { Upload, FileVideo, Film, Settings, Video } from 'lucide-react';

export default function UploadForm({ mode, setMode, onUpload }) {
    const [files, setFiles] = useState({ left: null, center: null, right: null });

    // Processing options state
    const [movingCamera, setMovingCamera] = useState(false);

    const handleFileChange = (position, e) => {
        if (e.target.files[0]) {
            setFiles({ ...files, [position]: e.target.files[0] });
        }
    };

    const isReady = () => {
        if (mode === '2cam') return files.left && files.right;
        return files.left && files.center && files.right;
    };

    const handleSubmit = () => {
        if (isReady()) {
            // Pass files and options to parent
            onUpload(files, { movingCamera });
        }
    };

    const ToggleSwitch = ({ enabled, onChange, label, icon: Icon, description }) => (
        <div
            onClick={() => onChange(!enabled)}
            className={`
                flex items-center gap-3 p-4 rounded-lg cursor-pointer transition-all duration-300 border
                ${enabled 
                    ? 'bg-black border-[#00D4FF]/50 shadow-[0_0_15px_rgba(0,212,255,0.15)]' 
                    : 'bg-black/60 border-[#222222] hover:border-[#333333]'}
            `}
        >
            <div className={`p-2 rounded-lg border ${enabled ? 'bg-[#00D4FF]/10 border-[#00D4FF]/30 text-[#00D4FF]' : 'bg-black border-[#222222] text-white/40'}`}>
                <Icon className="w-5 h-5" strokeWidth={1.5} />
            </div>
            <div className="flex-1">
                <p className={`font-medium text-sm ${enabled ? 'text-[#00D4FF]' : 'text-white/60'}`}>{label}</p>
                <p className="text-xs text-white/30">{description}</p>
            </div>
            <div className={`w-12 h-7 rounded-full p-1 transition-colors border ${enabled ? 'bg-[#00D4FF]/20 border-[#00D4FF]/50' : 'bg-black border-[#222222]'}`}>
                <div className={`w-5 h-5 rounded-full shadow-sm transition-transform duration-300 ${enabled ? 'translate-x-[20px] bg-[#00D4FF]' : 'translate-x-0 bg-[#333333]'}`} />
            </div>
        </div>
    );

    const UploadBox = ({ position, label }) => (
        <div className="flex flex-col gap-3">
            <label className="text-xs font-bold text-gradient-subtle uppercase tracking-widest pl-1 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#39FF14] animate-pulse"></span>
                {label} Camera
            </label>
            <div
                className={`
          relative group border rounded-lg p-10
          transition-all duration-300 ease-in-out
          flex flex-col items-center justify-center gap-4
          overflow-hidden
          ${files[position]
                        ? 'border-[#39FF14]/50 bg-[#39FF14]/5 shadow-[0_0_20px_rgba(57,255,20,0.1)]'
                        : 'border-[#222222] border-dashed hover:border-[#00D4FF]/50 hover:bg-[#00D4FF]/5'
                    }
        `}
            >
                <input
                    type="file"
                    accept="video/*"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    onChange={(e) => handleFileChange(position, e)}
                />

                {files[position] ? (
                    <>
                         <div className="absolute inset-0 bg-gradient-to-br from-[#39FF14]/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                        <div className="p-4 bg-black text-[#39FF14] rounded-lg border border-[#222222] relative z-0">
                            <FileVideo className="w-8 h-8" strokeWidth={1.5} />
                        </div>
                        <div className="text-center relative z-0">
                            <p className="font-semibold text-white truncate max-w-[180px]">
                                {files[position].name}
                            </p>
                            <p className="text-xs text-white/30 mt-1 font-mono">
                                {(files[position].size / (1024 * 1024)).toFixed(1)} MB
                            </p>
                        </div>
                    </>
                ) : (
                    <>
                        <div className="p-4 bg-black rounded-lg border border-[#222222] group-hover:border-[#00D4FF]/30 transition-all duration-300">
                            <Upload className="w-8 h-8 text-white/30 group-hover:text-[#00D4FF]" strokeWidth={1.5} />
                        </div>
                        <div className="text-center">
                            <p className="font-medium text-white/50 group-hover:text-white/70 transition-colors">Click to Upload</p>
                            <p className="text-xs text-white/20 mt-1">MP4, AVI, MOV</p>
                        </div>
                    </>
                )}
            </div>
        </div>
    );

    return (
        <div className="space-y-8 animate-fade-in glass-panel p-8 rounded-lg">
            <div className="text-center space-y-3">
                <h2 className="text-4xl font-bold tracking-tight">
                    <span className="text-gradient-white">Stitch Your</span> <span className="text-gradient">Panorama</span>
                </h2>
                <p className="text-gradient-subtle text-lg font-light">
                    Upload camera feeds to generate a seamless ultra-wide video.
                </p>
            </div>

            {/* Requirements Note */}
            <div className="bg-black/60 border border-[#222222] rounded-lg p-4">
                <div className="flex items-start gap-3">
                    <span className="w-2 h-2 rounded-full bg-[#00D4FF] mt-1.5 animate-pulse flex-shrink-0"></span>
                    <div className="text-sm text-white/70 space-y-2">
                        <p className="font-semibold text-[#00D4FF]">Input Requirements</p>
                        <div className="space-y-1 text-white/60 text-xs">
                            <p>• <span className="text-white/80">20-30% overlap</span> between adjacent camera views</p>
                            <p>• <span className="text-white/80">Same resolution</span> and frame rate for all videos</p>
                            <p>• <span className="text-white/80">Synchronized start</span> time for best results</p>
                            <p>• <span className="text-white/80">Static features</span> in overlap region improve alignment</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Mode Toggle - Cinematic style tabs */}
            <div className="flex justify-center">
                <div className="inline-flex p-1 bg-black border border-[#222222] rounded-lg">
                    <button
                        onClick={() => { setMode('2cam'); setFiles({ left: null, center: null, right: null }); }}
                        className={`px-8 py-2.5 rounded-md text-sm font-semibold transition-all duration-300 ${mode === '2cam'
                            ? 'bg-[#00D4FF] text-black shadow-[0_0_20px_rgba(0,212,255,0.3)]'
                            : 'text-white/40 hover:text-white hover:bg-white/5'
                            }`}
                    >
                        2 Cameras
                    </button>
                    <button
                        onClick={() => { setMode('3cam'); setFiles({ left: null, center: null, right: null }); }}
                        className={`px-8 py-2.5 rounded-md text-sm font-semibold transition-all duration-300 ${mode === '3cam'
                            ? 'bg-[#00D4FF] text-black shadow-[0_0_20px_rgba(0,212,255,0.3)]'
                            : 'text-white/40 hover:text-white hover:bg-white/5'
                            }`}
                    >
                        3 Cameras
                    </button>
                </div>
            </div>

            {/* Upload Grid - With cinematic bordered zones */}
            <div className={`grid gap-6 ${mode === '2cam' ? 'grid-cols-2' : 'grid-cols-3'}`}>
                <div className="animate-slide-up" style={{ animationDelay: '0.1s' }}>
                    <UploadBox position="left" label="Left" />
                </div>
                {mode === '3cam' && (
                    <div className="animate-slide-up" style={{ animationDelay: '0.2s' }}>
                        <UploadBox position="center" label="Center" />
                    </div>
                )}
                <div className="animate-slide-up" style={{ animationDelay: '0.3s' }}>
                    <UploadBox position="right" label="Right" />
                </div>
            </div>

            {/* Processing Options - Cinematic config section */}
            <div className="space-y-4 pt-6 border-t border-[#222222]">
                 <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-gradient-subtle pl-1">
                    <Settings className="w-3 h-3" strokeWidth={1.5} />
                    <span>Configuration</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <ToggleSwitch
                        enabled={movingCamera}
                        onChange={setMovingCamera}
                        label="Moving Camera Mode"
                        icon={Video}
                        description="Optimized for vehicle-mounted footage"
                    />
                </div>
            </div>

            {/* Action Button - Neon Lime for primary action */}
            <div className="flex justify-center pt-4">
                <button
                    onClick={handleSubmit}
                    disabled={!isReady()}
                    className={`
            px-10 py-4 rounded-lg font-bold text-lg flex items-center gap-3
            transition-all duration-300 transform group relative overflow-hidden
            ${isReady()
                            ? 'btn-neon-lime hover:scale-[1.02]'
                            : 'bg-black/60 text-white/20 cursor-not-allowed border border-[#222222]'
                        }
          `}
                >
                    <Film className={`w-5 h-5 ${isReady() ? 'animate-pulse' : ''}`} strokeWidth={1.5} />
                    Start Processing
                </button>
            </div>
        </div>
    );
}
