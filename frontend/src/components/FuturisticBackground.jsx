import React from 'react';

export default function FuturisticBackground() {
  return (
    <div className="absolute inset-0 w-full h-full bg-black overflow-hidden pointer-events-none perspective-[500px]">
      {/* Cinematic vignette overlay */}
      <div className="absolute top-0 left-0 w-full h-[30%] bg-gradient-to-b from-black via-black to-transparent z-10" />
      <div className="absolute bottom-0 left-0 w-full h-[70%] bg-gradient-to-t from-black via-transparent to-transparent z-10" />

      {/* Cinematic Grid Floor - Subtle cyan grid lines */}
      <div 
        className="absolute bottom-[-50%] left-[-50%] w-[200%] h-[100%] animate-grid-move origin-bottom"
        style={{
          background: `
            linear-gradient(to right, rgba(0, 212, 255, 0.03) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(0, 212, 255, 0.03) 1px, transparent 1px)
          `,
          backgroundSize: '4rem 4rem',
          transform: 'rotateX(60deg)'
        }}
      />

      {/* Subtle ambient particles with neon colors */}
      <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-1 h-1 bg-[#00D4FF]/20 rounded-full animate-pulse" />
          <div className="absolute top-3/4 left-3/4 w-1 h-1 bg-[#39FF14]/10 rounded-full animate-pulse" />
          <div className="absolute top-1/2 left-1/2 w-1 h-1 bg-[#00D4FF]/10 rounded-full animate-pulse" />
          <div className="absolute top-1/3 right-1/4 w-0.5 h-0.5 bg-[#00D4FF]/15 rounded-full animate-pulse" />
          <div className="absolute bottom-1/4 left-1/3 w-0.5 h-0.5 bg-[#39FF14]/15 rounded-full animate-pulse" />
      </div>

      {/* Horizontal scan line for studio feel */}
      <div className="absolute top-1/2 left-0 w-full h-px bg-gradient-to-r from-transparent via-[#00D4FF]/10 to-transparent" />
    </div>
  );
}
