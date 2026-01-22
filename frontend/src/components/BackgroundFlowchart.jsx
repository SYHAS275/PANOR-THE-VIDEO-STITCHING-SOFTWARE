import React from 'react';

export default function BackgroundFlowchart() {
  return (
    <div className="absolute inset-0 w-full h-full opacity-20 pointer-events-none overflow-hidden select-none flex items-center justify-center">
      <svg
        viewBox="0 0 1200 800"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full max-w-[1400px]"
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
          
          <linearGradient id="line-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.2" />
            <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.2" />
          </linearGradient>

          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
             <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="1"/>
          </pattern>
        </defs>

        {/* Background Grid */}
        <rect width="100%" height="100%" fill="url(#grid)" />

        {/* --- NODES --- */}
        
        {/* Input Nodes */}
        <g transform="translate(100, 200)">
           <rect x="0" y="0" width="120" height="80" rx="10" 
                 className="fill-black/40 stroke-white/20" strokeWidth="2" />
           <text x="60" y="45" textAnchor="middle" fill="#fff" className="text-sm font-mono opacity-80">CAM 1</text>
        </g>

        <g transform="translate(100, 360)">
           <rect x="0" y="0" width="120" height="80" rx="10" 
                 className="fill-black/40 stroke-white/20" strokeWidth="2" />
           <text x="60" y="45" textAnchor="middle" fill="#fff" className="text-sm font-mono opacity-80">CAM 2</text>
        </g>

        <g transform="translate(100, 520)">
           <rect x="0" y="0" width="120" height="80" rx="10" 
                 className="fill-black/40 stroke-white/20" strokeWidth="2" />
           <text x="60" y="45" textAnchor="middle" fill="#fff" className="text-sm font-mono opacity-80">CAM 3</text>
        </g>

        {/* Processing Node 1: Feature Extraction */}
        <g transform="translate(350, 360)">
           <rect x="0" y="-80" width="160" height="240" rx="16" 
                 className="fill-blue-900/10 stroke-blue-500/30" strokeWidth="2" filter="url(#glow)" />
           <text x="80" y="20" textAnchor="middle" fill="#93c5fd" className="text-sm font-bold tracking-widest">SIFT/ORB</text>
           <text x="80" y="50" textAnchor="middle" fill="#60a5fa" className="text-[10px] opacity-60">FEATURE</text>
           <text x="80" y="65" textAnchor="middle" fill="#60a5fa" className="text-[10px] opacity-60">EXTRACTION</text>
        </g>

        {/* Processing Node 2: Homography */}
        <g transform="translate(600, 360)">
           <circle cx="80" cy="40" r="70" 
                   className="fill-purple-900/10 stroke-purple-500/30" strokeWidth="2" filter="url(#glow)" />
           <text x="80" y="45" textAnchor="middle" fill="#d8b4fe" className="text-sm font-bold">WARPING</text>
        </g>

        {/* Output Node: Panorama */}
        <g transform="translate(850, 280)">
           <rect x="0" y="0" width="240" height="160" rx="10" 
                 className="fill-green-900/10 stroke-green-500/40" strokeWidth="3" filter="url(#glow)" />
            <text x="120" y="85" textAnchor="middle" fill="#86efac" className="text-xl font-bold tracking-widest">PANORAMA</text>
            <rect x="20" y="20" width="200" height="120" rx="4" className="stroke-green-500/10 fill-none" strokeDasharray="4 4" />
        </g>


        {/* --- CONNECTIONS --- */}
        
        {/* Cam 1 to Feature */}
        <path d="M 220 240 C 280 240, 280 300, 350 300" stroke="url(#line-gradient)" strokeWidth="2" fill="none" opacity="0.3" />

        {/* Cam 2 to Feature */}
        <path d="M 220 400 L 350 400" stroke="url(#line-gradient)" strokeWidth="2" fill="none" opacity="0.3" />

        {/* Cam 3 to Feature */}
        <path d="M 220 560 C 280 560, 280 500, 350 500" stroke="url(#line-gradient)" strokeWidth="2" fill="none" opacity="0.3" />

        {/* Feature to Warping */}
        <path d="M 510 400 L 600 400" stroke="white" strokeOpacity="0.1" strokeWidth="3" fill="none" />

        {/* Warping to Panorama */}
        <path d="M 750 400 L 850 360" stroke="white" strokeOpacity="0.1" strokeWidth="4" fill="none" />
        
        
        {/* --- GLOWING LIVE ANIMATION --- */}
        
        {/* The Path Track (faint background line for the flow) */}
        <path d="M 160 400 L 350 400 L 430 360 C 430 360 480 360 510 400 L 600 400 L 750 400 L 850 360 L 970 360" 
              stroke="white" strokeWidth="2" fill="none" opacity="0.1" strokeLinecap="round" />

        {/* Primary Glowing Line - Using CSS Animation Class */}
        <path d="M 160 400 L 350 400 L 430 360 C 430 360 480 360 510 400 L 600 400 L 750 400 L 850 360 L 970 360" 
              stroke="#60a5fa" 
              strokeWidth="8" 
              fill="none" 
              filter="url(#glow)"
              strokeLinecap="round"
              className="animate-dash-flow" 
        />

        {/* Secondary Glowing Line (Delayed) - Using CSS Animation Class */}
         <path d="M 160 400 L 350 400 L 430 360 C 430 360 480 360 510 400 L 600 400 L 750 400 L 850 360 L 970 360" 
              stroke="#a855f7" 
              strokeWidth="6" 
              fill="none" 
              filter="url(#glow)" 
              opacity="0.8"
              strokeLinecap="round"
              className="animate-dash-flow-delayed"
        />

      </svg>
    </div>
  );
}
