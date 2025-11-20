import React, { useState, useEffect } from 'react';
import { Bot, X, Loader2, Sparkles } from 'lucide-react';
import { getTutorExplanation } from '../services/geminiService';
import { TutorContext } from '../types';

interface GeminiTutorProps {
  context: TutorContext;
  isOpen: boolean;
  onClose: () => void;
}

export const GeminiTutor: React.FC<GeminiTutorProps> = ({ context, isOpen, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [explanation, setExplanation] = useState<string>("Initialize the system to receive guidance.");

  useEffect(() => {
    if (isOpen) {
      handleAskAI();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, context.gameMode]); // Re-trigger on open or mode change

  const handleAskAI = async () => {
    setLoading(true);
    const text = await getTutorExplanation(context);
    setExplanation(text);
    setLoading(false);
  };

  if (!isOpen) return null;

  return (
    <div className="absolute top-4 right-4 w-80 bg-panel-bg border border-neon-blue/30 rounded-xl shadow-2xl backdrop-blur-md z-50 overflow-hidden flex flex-col">
      <div className="p-3 bg-slate-900/50 border-b border-white/10 flex justify-between items-center">
        <div className="flex items-center gap-2 text-neon-blue">
          <Bot size={20} />
          <span className="font-mono font-bold text-sm tracking-wider">AI TUTOR</span>
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
          <X size={18} />
        </button>
      </div>

      <div className="p-4 min-h-[150px] text-sm text-slate-300 leading-relaxed">
        {loading ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-neon-blue animate-pulse">
            <Loader2 className="animate-spin" size={24} />
            <span className="text-xs font-mono">ANALYZING GAME STATE...</span>
          </div>
        ) : (
          <div className="prose prose-invert prose-sm max-w-none">
             <p>{explanation}</p>
          </div>
        )}
      </div>

      <div className="p-3 border-t border-white/10 bg-slate-900/30">
        <button
          onClick={handleAskAI}
          disabled={loading}
          className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-bold uppercase tracking-widest flex items-center justify-center gap-2 transition-all disabled:opacity-50"
        >
          <Sparkles size={14} />
          Analysis Update
        </button>
      </div>
    </div>
  );
};