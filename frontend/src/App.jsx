import React, { useState, useEffect, lazy, Suspense } from 'react';
import axios from 'axios';
import { Upload, Activity, CheckCircle, AlertCircle, LogOut, User } from 'lucide-react';
import UploadForm from './components/UploadForm';
import ProcessingStatus from './components/ProcessingStatus';
import ResultView from './components/ResultView';
import FuturisticBackground from './components/FuturisticBackground';
import AuthModal from './components/AuthModal';
const MagneticSnapHero = lazy(() => import('./components/MagneticSnapHero'));
const LiveAvatarHero = lazy(() => import('./components/LiveAvatarHero'));

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

function App() {
  const [step, setStep] = useState('upload'); // upload, processing, result, error
  const [mode, setMode] = useState('2cam'); // 2cam, 3cam
  const [jobId, setJobId] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [resultFile, setResultFile] = useState(null);
  
  // Auth state
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [showAuthModal, setShowAuthModal] = useState(false);

  // Check for existing auth on mount
  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');
    
    if (savedToken && savedUser) {
      setToken(savedToken);
      setUser(JSON.parse(savedUser));
    }
  }, []);

  const handleAuthSuccess = (userData, authToken) => {
    setUser(userData);
    setToken(authToken);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setToken(null);
    setStep('upload');
    setJobId(null);
    setResultFile(null);
  };

  const handleUpload = async (files, options = {}) => {
    // files is { left: File, right: File, center: File }
    // options is { movingCamera, enableDetection, useTimestamps }
    const formData = new FormData();

    // Append files in order
    if (files.left) formData.append('files', files.left);
    if (mode === '3cam' && files.center) formData.append('files', files.center);
    if (files.right) formData.append('files', files.right);

    // Build query params
    const params = new URLSearchParams({
      mode: mode,
      moving_camera: options.movingCamera ?? true,
      enable_detection: options.enableDetection ?? false,
      use_timestamps: options.useTimestamps ?? false
    });

    try {
      setStep('processing');
      const headers = { 'Content-Type': 'multipart/form-data' };
      
      // Add auth header if logged in
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
      
      const response = await axios.post(`${API_BASE}/upload?${params.toString()}`, formData, { headers });
      setJobId(response.data.job_id);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.response?.data?.detail || "Upload failed");
      setStep('error');
    }
  };

  const reset = () => {
    setStep('upload');
    setJobId(null);
    setErrorMsg('');
    setResultFile(null);
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col font-sans relative overflow-hidden">
      {/* Background Effects - Subtle grid for cinematic feel */}
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <FuturisticBackground />
        {/* Scanline overlay for studio effect */}
        <div className="absolute inset-0 scanline-overlay opacity-30"></div>
      </div>

      {/* Header - Cinematic Pro with thin borders */}
      <header className="relative z-10 border-b border-[#222222] bg-black/95 backdrop-blur-sm px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative group">
            {/* Ambient glow behind logo */}
            <div className="absolute inset-[-4px] bg-gradient-to-r from-[#00D4FF]/40 via-[#a855f7]/40 to-[#f59e0b]/40 rounded-full blur-xl opacity-70 group-hover:opacity-100 transition-opacity"></div>
            
            {/* Custom Logo */}
            <img 
              src="/logo.jpg" 
              alt="PANOR Logo" 
              className="w-10 h-10 relative z-10 rounded-lg object-cover"
            />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-gradient-white">PANOR</h1>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="text-xs font-medium text-[#00D4FF]/70 uppercase tracking-widest flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#39FF14] animate-pulse"></span>
            Video Stitcher
          </div>
          
          {/* Auth buttons */}
          {user ? (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#00D4FF]/10 border border-[#00D4FF]/30">
                <User className="w-4 h-4 text-[#00D4FF]" />
                <span className="text-sm text-[#00D4FF]">{user.name}</span>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[#222222] hover:border-red-500/50 text-white/60 hover:text-red-400 transition-all"
              >
                <LogOut className="w-4 h-4" />
                <span className="text-sm">Logout</span>
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowAuthModal(true)}
              className="px-4 py-2 rounded-lg bg-[#00D4FF] text-black font-semibold text-sm hover:bg-[#00D4FF]/90 transition-all shadow-[0_0_15px_rgba(0,212,255,0.3)]"
            >
              Sign In
            </button>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center p-6">
        <div className="w-full max-w-4xl">

          {/* Show login prompt if not authenticated */}
          {!user ? (
            <div className="text-center space-y-4 animate-fade-in -mt-32">
              {/* Title at the top */}
              <h2 className="text-4xl font-bold tracking-tight text-gradient-white uppercase">WELCOME TO PANOR</h2>
              
              <Suspense fallback={<div className="w-full max-w-3xl mx-auto h-[160px] rounded-2xl border border-[#222222] bg-black/40" />}>
                <LiveAvatarHero />
              </Suspense>
              
              <div className="glass-panel p-6 rounded-lg space-y-4">
                <p className="text-white/50 max-w-md mx-auto">
                  Sign in or create an account to start stitching your panoramic videos with multi-camera stitching technology.
                </p>
                <button
                  onClick={() => setShowAuthModal(true)}
                  className="mt-2 px-8 py-3 btn-neon-blue rounded-lg font-semibold transition-all"
                >
                  Get Started
                </button>
              </div>
            </div>
          ) : (
            <>
              {step === 'upload' && (
                <UploadForm mode={mode} setMode={setMode} onUpload={handleUpload} />
              )}

              {step === 'processing' && jobId && (
                <ProcessingStatus
                  jobId={jobId}
                  apiBase={API_BASE}
                  onComplete={(filename) => {
                    setResultFile(filename);
                    setStep('result');
                  }}
                  onError={(msg) => {
                    setErrorMsg(msg);
                    setStep('error');
                  }}
                />
              )}

              {step === 'result' && resultFile && (
                <ResultView
                  filename={resultFile}
                  apiBase={API_BASE}
                  jobId={jobId}
                  onReset={reset}
                />
              )}

              {step === 'error' && (
                <div className="text-center space-y-4 glass-panel p-8 rounded-lg animate-fade-in">
                  <div className="inline-flex p-4 rounded-lg bg-red-500/10 text-red-500 mb-4 border border-[#222222]">
                    <AlertCircle className="w-12 h-12" strokeWidth={1.5} />
                  </div>
                  <h2 className="text-2xl font-bold text-white">Something went wrong</h2>
                  <p className="text-white/50">{errorMsg}</p>
                  <button
                    onClick={reset}
                    className="mt-6 px-8 py-3 btn-neon-blue rounded-lg font-semibold transition-all"
                  >
                    Try Again
                  </button>
                </div>
              )}
            </>
          )}

        </div>
      </main>

      {/* Auth Modal */}
      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onAuthSuccess={handleAuthSuccess}
      />
    </div>
  );
}

export default App;
