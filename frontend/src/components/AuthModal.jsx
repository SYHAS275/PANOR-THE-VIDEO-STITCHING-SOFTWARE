import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Mail, Lock, User, LogIn, UserPlus, AlertCircle, Loader2, Sparkles, Film, Layers, ShieldCheck, Wand2 } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

// Floating orb component for background effects
const FloatingOrb = ({ delay, duration, size, color, startX, startY }) => (
  <motion.div
    className="absolute rounded-full blur-xl pointer-events-none"
    style={{
      width: size,
      height: size,
      background: color,
      left: startX,
      top: startY,
    }}
    animate={{
      x: [0, 30, -20, 10, 0],
      y: [0, -30, 20, -10, 0],
      scale: [1, 1.2, 0.9, 1.1, 1],
      opacity: [0.3, 0.5, 0.4, 0.6, 0.3],
    }}
    transition={{
      duration: duration,
      delay: delay,
      repeat: Infinity,
      ease: "easeInOut",
    }}
  />
);

// Animated particle component
const Particle = ({ delay }) => (
  <motion.div
    className="absolute w-1 h-1 bg-[#00D4FF] rounded-full"
    style={{
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
    }}
    animate={{
      y: [0, -100],
      opacity: [0, 1, 0],
      scale: [0, 1, 0],
    }}
    transition={{
      duration: 3,
      delay: delay,
      repeat: Infinity,
      ease: "easeOut",
    }}
  />
);

export default function AuthModal({ isOpen, onClose, onAuthSuccess }) {
  const [isLogin, setIsLogin] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [focusedField, setFocusedField] = useState(null);
  
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: ''
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/signup';
      const body = isLogin 
        ? { email: formData.email, password: formData.password }
        : { name: formData.name, email: formData.email, password: formData.password };
      
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      let data = null;
      try {
        data = await response.json();
      } catch (_) {
        data = null;
      }

      if (!response.ok) {
        throw new Error((data && data.detail) || 'Authentication failed');
      }

      // Store token and user info - user info is now included in the response!
      localStorage.setItem('token', data.access_token);
      
      // Use user info directly from the signup/login response
      if (data.user) {
        localStorage.setItem('user', JSON.stringify(data.user));
        onAuthSuccess(data.user, data.access_token);
        onClose();
      } else {
        // Fallback: try to get user info from /auth/me (for backward compatibility)
        const userResponse = await fetch(`${API_BASE}/auth/me`, {
          headers: { 'Authorization': `Bearer ${data.access_token}` }
        });
        
        if (userResponse.ok) {
          const userData = await userResponse.json();
          localStorage.setItem('user', JSON.stringify(userData));
          onAuthSuccess(userData, data.access_token);
          onClose();
        } else {
          throw new Error('Failed to retrieve user information. Please try again.');
        }
      }
    } catch (err) {
      const msg =
        (err && typeof err.message === 'string' && err.message.toLowerCase().includes('failed to fetch'))
          ? `Cannot reach the server. Please check your connection or server URL (${API_BASE}).`
          : err.message || 'Authentication failed';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  const switchMode = () => {
    setIsLogin(!isLogin);
    setError('');
    setFormData({ name: '', email: '', password: '' });
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
      >
        {/* Backdrop with gradient */}
        <motion.div 
          className="absolute inset-0 bg-black/90 backdrop-blur-md"
          onClick={onClose}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {/* Animated gradient overlay */}
          <div className="absolute inset-0 bg-gradient-to-br from-[#00D4FF]/10 via-transparent to-[#a855f7]/10" />
          
          {/* Floating orbs in backdrop */}
          <FloatingOrb delay={0} duration={8} size="300px" color="radial-gradient(circle, rgba(0,212,255,0.15) 0%, transparent 70%)" startX="-10%" startY="20%" />
          <FloatingOrb delay={2} duration={10} size="400px" color="radial-gradient(circle, rgba(168,85,247,0.12) 0%, transparent 70%)" startX="60%" startY="-10%" />
          <FloatingOrb delay={4} duration={12} size="250px" color="radial-gradient(circle, rgba(57,255,20,0.1) 0%, transparent 70%)" startX="70%" startY="60%" />
        </motion.div>
        
        {/* Modal */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.9, opacity: 0, y: 20 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="relative w-full max-w-md overflow-hidden"
        >
          {/* Animated gradient border */}
          <div className="absolute -inset-[1px] bg-gradient-to-r from-[#00D4FF] via-[#a855f7] to-[#00D4FF] rounded-2xl opacity-50 blur-sm animate-pulse" />
          <div className="absolute -inset-[1px] bg-gradient-to-r from-[#00D4FF] via-[#a855f7] to-[#00D4FF] rounded-2xl" 
            style={{
              backgroundSize: '200% 200%',
              animation: 'gradient-shift 3s ease infinite',
            }}
          />
          
          {/* Modal content */}
          <div className="relative bg-black rounded-2xl border border-[#222222] shadow-2xl overflow-hidden">
            {/* Floating particles inside modal */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              {[...Array(6)].map((_, i) => (
                <Particle key={i} delay={i * 0.5} />
              ))}
            </div>
            
            {/* Inner glow effect */}
            <div className="absolute inset-0 bg-gradient-to-br from-[#00D4FF]/5 via-transparent to-[#a855f7]/5 pointer-events-none" />
            
            {/* Header */}
            <div className="relative px-8 pt-8 pb-6 border-b border-[#222222]/50">
              <motion.button
                onClick={onClose}
                whileHover={{ scale: 1.1, rotate: 90 }}
                whileTap={{ scale: 0.9 }}
                className="absolute top-4 right-4 p-2 text-white/40 hover:text-white transition-colors rounded-full hover:bg-white/10"
              >
                <X className="w-5 h-5" />
              </motion.button>
              
              <div className="flex items-center gap-4">
                {/* Animated icon container */}
                <motion.div 
                  className="relative p-4 rounded-xl bg-gradient-to-br from-[#00D4FF]/20 to-[#a855f7]/20 border border-[#00D4FF]/30"
                  animate={{
                    boxShadow: [
                      '0 0 20px rgba(0,212,255,0.2)',
                      '0 0 40px rgba(0,212,255,0.4)',
                      '0 0 20px rgba(0,212,255,0.2)',
                    ],
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  <motion.div
                    key={isLogin ? 'login' : 'signup'}
                    initial={{ rotate: -180, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    transition={{ type: "spring", damping: 15 }}
                  >
                    {isLogin ? (
                      <LogIn className="w-7 h-7 text-[#00D4FF]" />
                    ) : (
                      <UserPlus className="w-7 h-7 text-[#00D4FF]" />
                    )}
                  </motion.div>
                  
                  {/* Sparkle effect */}
                  <motion.div
                    className="absolute -top-1 -right-1"
                    animate={{
                      scale: [1, 1.2, 1],
                      opacity: [0.5, 1, 0.5],
                    }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <Sparkles className="w-4 h-4 text-[#00D4FF]" />
                  </motion.div>
                </motion.div>
                
                <div>
                  <motion.h2 
                    key={isLogin ? 'login-title' : 'signup-title'}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="text-2xl font-bold text-white"
                  >
                    {isLogin ? 'Welcome Back' : 'Create Account'}
                  </motion.h2>
                  <motion.p 
                    key={isLogin ? 'login-desc' : 'signup-desc'}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                    className="text-sm text-white/50 mt-1"
                  >
                    {isLogin ? 'Sign in to access your workspace' : 'Join to start stitching videos'}
                  </motion.p>
                </div>
              </div>
            </div>
            
            {/* Form */}
            <form onSubmit={handleSubmit} className="relative p-8 space-y-5">
              {/* Error message */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10, height: 0 }}
                    animate={{ opacity: 1, y: 0, height: 'auto' }}
                    exit={{ opacity: 0, y: -10, height: 0 }}
                    className="flex items-center gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm"
                  >
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <span>{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>
              
              {/* Name field (signup only) */}
              <AnimatePresence>
                {!isLogin && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                  >
                    <label className="block text-xs font-medium text-white/60 uppercase tracking-wider mb-2">
                      Full Name
                    </label>
                    <div className="relative group">
                      <div className={`absolute inset-0 rounded-xl transition-all duration-300 ${
                        focusedField === 'name' 
                          ? 'bg-gradient-to-r from-[#00D4FF]/20 to-[#a855f7]/20 blur-md' 
                          : ''
                      }`} />
                      <div className="relative flex items-center">
                        <User className={`absolute left-4 w-5 h-5 transition-colors duration-300 ${
                          focusedField === 'name' ? 'text-[#00D4FF]' : 'text-white/30'
                        }`} />
                        <input
                          type="text"
                          name="name"
                          value={formData.name}
                          onChange={handleChange}
                          onFocus={() => setFocusedField('name')}
                          onBlur={() => setFocusedField(null)}
                          required={!isLogin}
                          placeholder="John Doe"
                          className="w-full pl-12 pr-4 py-4 bg-black/50 border border-[#333333] rounded-xl text-white placeholder-white/30 focus:border-[#00D4FF] focus:outline-none focus:ring-2 focus:ring-[#00D4FF]/20 transition-all duration-300"
                        />
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              
              {/* Email field */}
              <div>
                <label className="block text-xs font-medium text-white/60 uppercase tracking-wider mb-2">
                  Email Address
                </label>
                <div className="relative group">
                  <div className={`absolute inset-0 rounded-xl transition-all duration-300 ${
                    focusedField === 'email' 
                      ? 'bg-gradient-to-r from-[#00D4FF]/20 to-[#a855f7]/20 blur-md' 
                      : ''
                  }`} />
                  <div className="relative flex items-center">
                    <Mail className={`absolute left-4 w-5 h-5 transition-colors duration-300 ${
                      focusedField === 'email' ? 'text-[#00D4FF]' : 'text-white/30'
                    }`} />
                    <input
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      onFocus={() => setFocusedField('email')}
                      onBlur={() => setFocusedField(null)}
                      required
                      placeholder="you@example.com"
                      className="w-full pl-12 pr-4 py-4 bg-black/50 border border-[#333333] rounded-xl text-white placeholder-white/30 focus:border-[#00D4FF] focus:outline-none focus:ring-2 focus:ring-[#00D4FF]/20 transition-all duration-300"
                    />
                  </div>
                </div>
              </div>
              
              {/* Password field */}
              <div>
                <label className="block text-xs font-medium text-white/60 uppercase tracking-wider mb-2">
                  Password
                </label>
                <div className="relative group">
                  <div className={`absolute inset-0 rounded-xl transition-all duration-300 ${
                    focusedField === 'password' 
                      ? 'bg-gradient-to-r from-[#00D4FF]/20 to-[#a855f7]/20 blur-md' 
                      : ''
                  }`} />
                  <div className="relative flex items-center">
                    <Lock className={`absolute left-4 w-5 h-5 transition-colors duration-300 ${
                      focusedField === 'password' ? 'text-[#00D4FF]' : 'text-white/30'
                    }`} />
                    <input
                      type="password"
                      name="password"
                      value={formData.password}
                      onChange={handleChange}
                      onFocus={() => setFocusedField('password')}
                      onBlur={() => setFocusedField(null)}
                      required
                      minLength={6}
                      placeholder="••••••••"
                      className="w-full pl-12 pr-4 py-4 bg-black/50 border border-[#333333] rounded-xl text-white placeholder-white/30 focus:border-[#00D4FF] focus:outline-none focus:ring-2 focus:ring-[#00D4FF]/20 transition-all duration-300"
                    />
                  </div>
                </div>
              </div>

              <AnimatePresence>
                {!isLogin && (
                  <motion.div
                    initial={{ opacity: 0, y: 8, height: 0 }}
                    animate={{ opacity: 1, y: 0, height: 'auto' }}
                    exit={{ opacity: 0, y: 8, height: 0 }}
                    transition={{ duration: 0.25 }}
                    className="rounded-xl border border-[#222222] bg-black/40 overflow-hidden"
                  >
                    <div className="px-4 py-3 border-b border-[#222222] flex items-center gap-2">
                      <Wand2 className="w-4 h-4 text-[#00D4FF]" />
                      <div className="text-xs font-bold uppercase tracking-widest text-white/70">Built for stitching</div>
                    </div>
                    <div className="px-4 py-4 space-y-3">
                      <div className="flex items-start gap-3">
                        <div className="p-2 rounded-lg bg-[#00D4FF]/10 border border-[#00D4FF]/20 text-[#00D4FF]">
                          <Film className="w-4 h-4" />
                        </div>
                        <div className="min-w-0">
                          <div className="text-sm font-semibold text-white/80">2 or 3 camera stitching</div>
                          <div className="text-xs text-white/40">Preview stitched output and compare inputs side by side.</div>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <div className="p-2 rounded-lg bg-[#a855f7]/10 border border-[#a855f7]/20 text-[#a855f7]">
                          <Layers className="w-4 h-4" />
                        </div>
                        <div className="min-w-0">
                          <div className="text-sm font-semibold text-white/80">Better seams</div>
                          <div className="text-xs text-white/40">Best results with 20–30% overlap and similar FPS.</div>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <div className="p-2 rounded-lg bg-[#39FF14]/10 border border-[#39FF14]/20 text-[#39FF14]">
                          <ShieldCheck className="w-4 h-4" />
                        </div>
                        <div className="min-w-0">
                          <div className="text-sm font-semibold text-white/80">Private workspace</div>
                          <div className="text-xs text-white/40">Sign in to manage your stitching sessions from one place.</div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              
              {/* Submit button */}
              <motion.button
                type="submit"
                disabled={isLoading}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                className="relative w-full py-4 px-6 rounded-xl font-semibold text-black overflow-hidden disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 group"
              >
                {/* Button background gradient */}
                <div className="absolute inset-0 bg-gradient-to-r from-[#00D4FF] via-[#00E5FF] to-[#00D4FF] bg-[length:200%_100%] group-hover:animate-gradient-x" />
                
                {/* Button glow */}
                <div className="absolute inset-0 bg-[#00D4FF] blur-lg opacity-40 group-hover:opacity-60 transition-opacity" />
                
                {/* Button content */}
                <span className="relative flex items-center justify-center gap-2">
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      {isLogin ? 'Signing in...' : 'Creating account...'}
                    </>
                  ) : (
                    <>
                      {isLogin ? <LogIn className="w-5 h-5" /> : <UserPlus className="w-5 h-5" />}
                      {isLogin ? 'Sign In' : 'Create Account'}
                    </>
                  )}
                </span>
              </motion.button>
            </form>
            
            {/* Footer - Switch mode */}
            <div className="relative px-8 py-6 border-t border-[#222222]/50 text-center bg-gradient-to-t from-[#00D4FF]/5 to-transparent">
              <p className="text-sm text-white/50">
                {isLogin ? "Don't have an account?" : "Already have an account?"}{' '}
                <motion.button
                  type="button"
                  onClick={switchMode}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="text-[#00D4FF] hover:text-[#00D4FF]/80 font-medium transition-colors relative group"
                >
                  {isLogin ? 'Sign up' : 'Sign in'}
                  <span className="absolute bottom-0 left-0 w-0 h-[1px] bg-[#00D4FF] group-hover:w-full transition-all duration-300" />
                </motion.button>
              </p>
            </div>
          </div>
        </motion.div>
      </motion.div>
      
      {/* Keyframes for gradient animation */}
      <style>{`
        @keyframes gradient-shift {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        @keyframes gradient-x {
          0% { background-position: 0% 50%; }
          100% { background-position: 200% 50%; }
        }
        .animate-gradient-x {
          animation: gradient-x 2s linear infinite;
        }
      `}</style>
    </AnimatePresence>
  );
}
