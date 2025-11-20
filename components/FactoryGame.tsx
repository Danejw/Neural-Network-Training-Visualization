import React, { useState, useEffect } from 'react';
import { ArrowRight, AlertTriangle, Settings, CheckCircle } from 'lucide-react';
import { TutorContext, GameMode } from '../types';

interface FactoryGameProps {
  onUpdateContext: (ctx: TutorContext) => void;
}

const TARGET_OUTPUT = 75;

export const FactoryGame: React.FC<FactoryGameProps> = ({ onUpdateContext }) => {
  const [input, setInput] = useState(10);
  const [weight, setWeight] = useState(1.0);
  const [bias, setBias] = useState(0);
  const [output, setOutput] = useState(0);
  const [error, setError] = useState(0);
  const [mode, setMode] = useState<'FORWARD' | 'DEBUG'>('FORWARD');

  // y = mx + b
  const calculateOutput = () => {
    return (input * weight) + bias;
  };

  useEffect(() => {
    const res = calculateOutput();
    setOutput(res);
    setError(Math.abs(TARGET_OUTPUT - res));
  }, [input, weight, bias]);

  useEffect(() => {
    onUpdateContext({
      gameMode: GameMode.FACTORY,
      currentStats: { input, weight, bias, output, target: TARGET_OUTPUT, error }
    });
  }, [output, mode, onUpdateContext]); // Dependencies for context update

  const isWin = error < 2;

  // Visual helpers
  const pipeWidth = Math.max(2, Math.min(20, 5 * weight)); // Visual thickness
  const flowColor = error > 20 ? 'stroke-red-500' : error > 10 ? 'stroke-yellow-500' : 'stroke-neon-green';

  return (
    <div className="h-full flex flex-col p-6 relative">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-slate-800/30 to-dark-bg -z-10" />
      
      <header className="mb-6 flex justify-between items-center border-b border-white/10 pb-4">
        <div>
          <h2 className="text-2xl font-mono font-bold text-orange-500 flex items-center gap-2">
            <Settings className="animate-spin-slow" />
            THE NEURAL FACTORY
          </h2>
          <p className="text-slate-400 text-sm mt-1">Adjust the heavy valve (Weight) and piston (Bias) to match production targets.</p>
        </div>
        <div className="text-right font-mono">
          <div className="text-xs text-slate-500">TARGET OUTPUT</div>
          <div className="text-3xl text-white">{TARGET_OUTPUT}</div>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-8 items-center">
        
        {/* Controls */}
        <div className="bg-panel-bg p-6 rounded-xl border border-white/10 space-y-6 shadow-lg">
          <div>
            <label className="block text-xs font-bold text-slate-400 mb-2 uppercase tracking-wider">
              Raw Material (Input)
            </label>
            <input 
              type="range" min="1" max="20" value={input} 
              onChange={(e) => setInput(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <div className="text-right font-mono text-blue-400">{input} units</div>
          </div>

          <div>
            <label className="block text-xs font-bold text-orange-400 mb-2 uppercase tracking-wider flex justify-between">
              <span>Valve Tightness (Weight)</span>
              {mode === 'DEBUG' && <span className="text-red-500 animate-pulse">HIGH HEAT DETECTED</span>}
            </label>
            <input 
              type="range" min="0.1" max="10.0" step="0.1" value={weight} 
              onChange={(e) => setWeight(Number(e.target.value))}
              className={`w-full ${mode === 'DEBUG' ? 'accent-red-500' : 'accent-orange-500'}`}
            />
            <div className="text-right font-mono text-orange-400">x{weight.toFixed(1)}</div>
          </div>

          <div>
            <label className="block text-xs font-bold text-purple-400 mb-2 uppercase tracking-wider">
              Auxiliary Piston (Bias)
            </label>
            <input 
              type="range" min="-50" max="50" step="1" value={bias} 
              onChange={(e) => setBias(Number(e.target.value))}
              className="w-full accent-purple-500"
            />
            <div className="text-right font-mono text-purple-400">{bias > 0 ? '+' : ''}{bias}</div>
          </div>

          <button 
            onClick={() => setMode(mode === 'FORWARD' ? 'DEBUG' : 'FORWARD')}
            className={`w-full py-3 font-bold font-mono text-sm rounded transition-colors ${
              mode === 'DEBUG' 
              ? 'bg-red-500/20 text-red-400 border border-red-500' 
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            {mode === 'DEBUG' ? 'EXIT DEBUG MODE' : 'INSPECT ERROR (BACKPROP)'}
          </button>
        </div>

        {/* Visualization */}
        <div className="col-span-2 h-full relative bg-slate-900/50 rounded-xl border border-white/5 overflow-hidden flex items-center justify-center">
          {/* Background Grid */}
          <div className="absolute inset-0 opacity-10" style={{backgroundImage: 'radial-gradient(#fff 1px, transparent 1px)', backgroundSize: '20px 20px'}}></div>

          <svg viewBox="0 0 800 400" className="w-full h-full max-w-3xl">
             {/* Definitions */}
             <defs>
               <filter id="glow">
                 <feGaussianBlur stdDeviation="4.5" result="coloredBlur"/>
                 <feMerge>
                   <feMergeNode in="coloredBlur"/>
                   <feMergeNode in="SourceGraphic"/>
                 </feMerge>
               </filter>
             </defs>

             {/* Connection Line */}
             <line 
               x1="100" y1="200" 
               x2="400" y2="200" 
               strokeWidth={pipeWidth} 
               className={`${flowColor} transition-all duration-300`}
               strokeLinecap="round"
             />
              <line 
               x1="400" y1="200" 
               x2="700" y2="200" 
               strokeWidth={pipeWidth} 
               className={`${flowColor} transition-all duration-300`}
               strokeDasharray="10,5"
             />

             {/* Input Node */}
             <circle cx="100" cy="200" r="40" className="fill-slate-800 stroke-blue-500 stroke-2" />
             <text x="100" y="205" textAnchor="middle" className="fill-white font-mono font-bold text-lg">{input}</text>
             <text x="100" y="260" textAnchor="middle" className="fill-blue-400 font-mono text-xs">INPUT</text>

             {/* Neuron/Weight Node */}
             <g className="transition-transform duration-300" style={{ transformBox: 'fill-box', transformOrigin: 'center', transform: `scale(${1 + (weight/20)})` }}>
                <rect x="360" y="160" width="80" height="80" rx="10" className={`${mode === 'DEBUG' && !isWin ? 'fill-red-900 stroke-red-500' : 'fill-slate-800 stroke-orange-500'} stroke-2 transition-colors`} />
                <text x="400" y="205" textAnchor="middle" className="fill-white font-mono font-bold text-xl">x{weight}</text>
                <text x="400" y="145" textAnchor="middle" className="fill-orange-400 font-mono text-xs">WEIGHT</text>
             </g>
             
             {/* Bias Indicator */}
             <text x="400" y="270" textAnchor="middle" className="fill-purple-400 font-mono text-sm font-bold">{bias > 0 ? '+' : ''}{bias} (BIAS)</text>

             {/* Output Node */}
             <circle cx="700" cy="200" r="50" className={`fill-slate-800 stroke-2 ${isWin ? 'stroke-neon-green' : 'stroke-white'}`} />
             <text x="700" y="205" textAnchor="middle" className="fill-white font-mono font-bold text-2xl">{output.toFixed(0)}</text>
             <text x="700" y="275" textAnchor="middle" className="fill-slate-400 font-mono text-xs">ACTUAL</text>
             
             {/* Error Visualization (Backprop) */}
             {mode === 'DEBUG' && !isWin && (
               <g>
                 <path d="M 650 180 Q 550 100 440 160" fill="none" stroke="#ef4444" strokeWidth="3" strokeDasharray="5,5" className="animate-pulse" />
                 <text x="550" y="120" textAnchor="middle" className="fill-red-400 font-mono text-sm bg-black">ERROR SIGNAL</text>
               </g>
             )}
          </svg>

          {isWin && (
             <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-neon-green/20 border border-neon-green backdrop-blur px-8 py-4 rounded-full flex items-center gap-4 animate-bounce">
               <CheckCircle className="text-neon-green" size={32} />
               <span className="text-neon-green font-bold font-mono text-xl">OPTIMIZED</span>
             </div>
          )}
        </div>
      </div>
    </div>
  );
};