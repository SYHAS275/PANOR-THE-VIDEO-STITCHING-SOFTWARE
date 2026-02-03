import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export default function MagneticSnapHero({ children }) {
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReducedMotion(Boolean(media.matches));
    update();
    if (media.addEventListener) media.addEventListener('change', update);
    else media.addListener(update);
    return () => {
      if (media.removeEventListener) media.removeEventListener('change', update);
      else media.removeListener(update);
    };
  }, []);

  return (
    <div className="relative w-full max-w-3xl mx-auto">
      <div className="absolute inset-0 bg-gradient-to-r from-[#00D4FF]/10 via-[#a855f7]/10 to-[#39FF14]/10 blur-2xl rounded-3xl pointer-events-none" />
      <div className="relative border border-[#222222] bg-black/40 rounded-2xl overflow-hidden shadow-[0_0_40px_rgba(0,0,0,0.6)]">
        <div className="relative h-[240px] sm:h-[280px]">
          <motion.div
            className="absolute left-1/2 top-1/2 w-[210px] sm:w-[240px] h-[140px] sm:h-[160px] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-[#a855f7]/40 bg-gradient-to-br from-[#0a0a0a] via-[#111827] to-[#0a0a0a] shadow-[0_0_24px_rgba(168,85,247,0.15)]"
            animate={
              reducedMotion
                ? { x: 0, y: 0, rotate: 0, scale: 1 }
                : {
                    x: [0, 0, 0, 0],
                    y: [-45, 0, -6, 0],
                    rotate: [2, 0, 0, 0],
                    scale: [0.98, 1, 1.01, 1]
                  }
            }
            transition={{ duration: 2.8, repeat: reducedMotion ? 0 : Infinity, ease: 'easeInOut' }}
          >
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-tr from-[#a855f7]/10 via-transparent to-[#00D4FF]/10" />
            <div className="absolute left-4 top-4 h-2 w-12 rounded-full bg-white/10" />
            <div className="absolute left-4 top-8 h-2 w-20 rounded-full bg-white/5" />
            <div className="absolute right-4 bottom-4 h-9 w-9 rounded-xl bg-[#00D4FF]/15 border border-[#00D4FF]/25" />
          </motion.div>

          <motion.div
            className="absolute left-1/2 top-1/2 w-[190px] sm:w-[220px] h-[125px] sm:h-[145px] -translate-y-1/2 rounded-2xl border border-[#00D4FF]/45 bg-gradient-to-br from-[#070a0f] via-[#0b1220] to-[#070a0f] shadow-[0_0_24px_rgba(0,212,255,0.18)]"
            style={{ translateX: '-50%' }}
            animate={
              reducedMotion
                ? { x: -140, rotate: 0 }
                : {
                    x: [-220, -40, -12, -16, -14],
                    rotate: [-8, -1, 0, 0, 0]
                  }
            }
            transition={{ duration: 2.8, repeat: reducedMotion ? 0 : Infinity, ease: 'easeInOut' }}
          >
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-tr from-[#00D4FF]/12 via-transparent to-transparent" />
            <div className="absolute left-4 bottom-4 h-7 w-24 rounded-lg bg-white/5 border border-white/5" />
          </motion.div>

          <motion.div
            className="absolute left-1/2 top-1/2 w-[190px] sm:w-[220px] h-[125px] sm:h-[145px] -translate-y-1/2 rounded-2xl border border-[#39FF14]/35 bg-gradient-to-br from-[#070f0a] via-[#08150f] to-[#070f0a] shadow-[0_0_24px_rgba(57,255,20,0.12)]"
            style={{ translateX: '-50%' }}
            animate={
              reducedMotion
                ? { x: 140, rotate: 0 }
                : {
                    x: [220, 40, 12, 16, 14],
                    rotate: [8, 1, 0, 0, 0]
                  }
            }
            transition={{ duration: 2.8, repeat: reducedMotion ? 0 : Infinity, ease: 'easeInOut' }}
          >
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-tl from-[#39FF14]/10 via-transparent to-transparent" />
            <div className="absolute right-4 top-4 h-7 w-24 rounded-lg bg-white/5 border border-white/5" />
          </motion.div>

          <motion.div
            className="absolute left-1/2 top-1/2 w-[34px] h-[170px] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-gradient-to-b from-[#00D4FF]/0 via-[#00D4FF]/30 to-[#00D4FF]/0 blur-[2px]"
            animate={
              reducedMotion ? { opacity: 0.35, scaleX: 1 } : { opacity: [0, 0, 0.85, 0, 0], scaleX: [0.85, 0.9, 1.1, 0.9, 0.85] }
            }
            transition={{ duration: 2.8, repeat: reducedMotion ? 0 : Infinity, ease: 'easeInOut' }}
          />

          <div className="absolute inset-x-0 bottom-0 h-20 bg-gradient-to-t from-black/70 via-black/10 to-transparent pointer-events-none" />
          {children ? (
            <div className="absolute inset-0 z-20 flex items-center justify-center p-6 text-center">
              <div className="max-w-md w-full">
                {children}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
