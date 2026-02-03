import React, { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';

export default function LiveAvatarHero() {
  const [reducedMotion, setReducedMotion] = useState(false);
  const [text, setText] = useState('');

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

  const message = 'Hi! 👋';
  useEffect(() => {
    setText('');
    if (reducedMotion) {
      setText(message);
      return;
    }
    let i = 0;
    const id = setInterval(() => {
      i += 1;
      setText(message.slice(0, i));
      if (i >= message.length) clearInterval(id);
    }, 100);
    return () => clearInterval(id);
  }, [reducedMotion]);

  const floatAnim = useMemo(
    () =>
      reducedMotion
        ? {}
        : {
            y: [0, -5, 0],
            transition: { duration: 2.5, repeat: Infinity, ease: 'easeInOut' }
          },
    [reducedMotion]
  );

  return (
    <div className="relative w-full flex items-center justify-center py-8 pt-14">
      <motion.div className="relative" animate={floatAnim}>
        
        {/* Speech bubble */}
        <motion.div
          className="absolute -top-10 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-xl bg-black border border-[#00D4FF]/50 text-sm font-medium shadow-[0_0_15px_rgba(0,212,255,0.2)]"
          initial={reducedMotion ? false : { opacity: 0, y: 8, scale: 0.8 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.4, delay: 0.2, ease: 'backOut' }}
        >
          <span className="text-[#00D4FF]">{text}</span>
          <span 
            className={reducedMotion ? 'hidden' : 'inline-block w-[2px] ml-0.5 bg-[#00D4FF] align-middle'} 
            style={{ height: 12, animation: 'blink-caret 0.8s step-end infinite' }} 
          />
          <div className="absolute left-1/2 -bottom-1.5 -translate-x-1/2 w-2.5 h-2.5 bg-black border-b border-r border-[#00D4FF]/50 rotate-45" />
        </motion.div>

        {/* Small Robot Container */}
        <div className="relative w-[100px] h-[120px]">
          
          {/* Gradient glow behind robot */}
          <div className="absolute inset-[-8px] rounded-3xl bg-gradient-to-br from-[#00D4FF]/20 via-[#a855f7]/15 to-[#39FF14]/20 blur-xl opacity-60" />
          
          {/* Robot Body - Black with gradient border */}
          <motion.div 
            className="absolute left-1/2 top-[50px] -translate-x-1/2 w-[55px] h-[50px] rounded-2xl bg-black border border-transparent shadow-[0_0_0_1px_rgba(0,212,255,0.4)]"
            style={{
              background: 'linear-gradient(135deg, #0a0a0a, #000000, #050505)',
              boxShadow: '0 0 0 1px rgba(0,212,255,0.3), inset 0 1px 0 rgba(255,255,255,0.05), 0 4px 20px rgba(0,0,0,0.8)'
            }}
            animate={reducedMotion ? {} : { scale: [1, 1.02, 1] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
          >
            {/* Gradient edge effect */}
            <div className="absolute inset-0 rounded-2xl p-[1px] overflow-hidden">
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-[#00D4FF]/40 via-transparent to-[#a855f7]/40" />
            </div>
            
            {/* Chest light */}
            <motion.div
              className="absolute top-[50%] left-1/2 -translate-x-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-[#00D4FF]"
              style={{ boxShadow: '0 0 8px #00D4FF, 0 0 16px #00D4FF' }}
              animate={reducedMotion ? {} : { opacity: [1, 0.4, 1], scale: [1, 1.2, 1] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
            />
          </motion.div>

          {/* Robot Head - Black with gradient border */}
          <motion.div 
            className="absolute left-1/2 top-0 -translate-x-1/2 w-[48px] h-[44px] rounded-xl bg-black"
            style={{
              background: 'linear-gradient(135deg, #0a0a0a, #000000, #050505)',
              boxShadow: '0 0 0 1px rgba(0,212,255,0.3), inset 0 1px 0 rgba(255,255,255,0.05), 0 4px 15px rgba(0,0,0,0.8)'
            }}
            animate={reducedMotion ? {} : { rotate: [-1, 1, -1] }}
            transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
          >
            {/* Gradient edge effect */}
            <div className="absolute inset-0 rounded-xl p-[1px] overflow-hidden">
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-[#00D4FF]/50 via-transparent to-[#a855f7]/50" />
            </div>

            {/* Antenna */}
            <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-[2px] h-2 bg-gradient-to-t from-[#333] to-[#00D4FF]" />
            <motion.div 
              className="absolute -top-3.5 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-[#00D4FF]"
              style={{ boxShadow: '0 0 6px #00D4FF, 0 0 12px #00D4FF' }}
              animate={reducedMotion ? {} : { scale: [1, 1.3, 1], opacity: [1, 0.6, 1] }}
              transition={{ duration: 1, repeat: Infinity, ease: 'easeInOut' }}
            />

            {/* Eyes */}
            <div className="absolute top-[35%] left-1/2 -translate-x-1/2 flex gap-2.5">
              <motion.div 
                className="w-3 h-3 rounded-full bg-[#00D4FF]"
                style={{ boxShadow: '0 0 6px #00D4FF' }}
                animate={reducedMotion ? {} : { scaleY: [1, 0.1, 1] }}
                transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut', times: [0, 0.02, 0.04] }}
              />
              <motion.div 
                className="w-3 h-3 rounded-full bg-[#00D4FF]"
                style={{ boxShadow: '0 0 6px #00D4FF' }}
                animate={reducedMotion ? {} : { scaleY: [1, 0.1, 1] }}
                transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut', times: [0, 0.02, 0.04] }}
              />
            </div>

            {/* Smile */}
            <div className="absolute bottom-[20%] left-1/2 -translate-x-1/2 w-4 h-2 border-b-2 border-[#00D4FF] rounded-b-full" style={{ boxShadow: '0 2px 4px rgba(0,212,255,0.3)' }} />
          </motion.div>

          {/* Left Arm (static) */}
          <div 
            className="absolute left-[8px] top-[55px] w-[10px] h-[28px] rounded-full bg-black"
            style={{
              background: 'linear-gradient(135deg, #0a0a0a, #000000)',
              boxShadow: '0 0 0 1px rgba(0,212,255,0.25)'
            }}
          />

          {/* Right Arm (waving) */}
          <motion.div
            className="absolute right-[0px] top-[48px] origin-bottom"
            animate={reducedMotion ? {} : { rotate: [-15, 30, -15] }}
            transition={{ duration: 0.5, repeat: Infinity, ease: 'easeInOut' }}
          >
            <div 
              className="w-[10px] h-[28px] rounded-full bg-black"
              style={{
                background: 'linear-gradient(135deg, #0a0a0a, #000000)',
                boxShadow: '0 0 0 1px rgba(0,212,255,0.25)'
              }}
            />
            {/* Hand */}
            <motion.div 
              className="absolute -top-0.5 left-1/2 -translate-x-1/2 w-3.5 h-3.5 rounded-full bg-black"
              style={{
                background: 'linear-gradient(135deg, #0a0a0a, #000000)',
                boxShadow: '0 0 0 1px rgba(0,212,255,0.3)'
              }}
              animate={reducedMotion ? {} : { rotate: [-25, 25, -25] }}
              transition={{ duration: 0.25, repeat: Infinity, ease: 'easeInOut' }}
            />
          </motion.div>

          {/* Legs */}
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 flex gap-2">
            <div 
              className="w-[12px] h-[16px] rounded-b-lg bg-black"
              style={{
                background: 'linear-gradient(180deg, #0a0a0a, #000000)',
                boxShadow: '0 0 0 1px rgba(0,212,255,0.2)'
              }}
            />
            <div 
              className="w-[12px] h-[16px] rounded-b-lg bg-black"
              style={{
                background: 'linear-gradient(180deg, #0a0a0a, #000000)',
                boxShadow: '0 0 0 1px rgba(0,212,255,0.2)'
              }}
            />
          </div>
        </div>

        {/* Shadow under robot */}
        <motion.div 
          className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-[60px] h-[8px] rounded-full bg-[#00D4FF]/15 blur-md"
          animate={reducedMotion ? {} : { scaleX: [1, 1.15, 1], opacity: [0.2, 0.35, 0.2] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
        />
      </motion.div>

      <style>{`
        @keyframes blink-caret {
          0%, 100% { opacity: 0; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
