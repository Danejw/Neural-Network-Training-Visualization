import React, { useState, useEffect, useRef } from 'react';
import { RefreshCw, Sparkles } from 'lucide-react';
import { TutorContext, GameMode } from '../types';

interface PrismGameProps {
  onUpdateContext: (ctx: TutorContext) => void;
}

export const PrismGame: React.FC<PrismGameProps> = ({ onUpdateContext }) => {
  const [rotation, setRotation] = useState(0); // In degrees
  const [targetAngle, setTargetAngle] = useState(45);
  
  // Vector logic
  const length = 150;
  const inputVector = { x: 150, y: 0 }; // Start pointing right
  
  const rad = (deg: number) => (deg * Math.PI) / 180;
  
  // Rotate vector by matrix [[cos, -sin], [sin, cos]]
  const outputVector = {
    x: inputVector.x * Math.cos(rad(rotation)) - inputVector.y * Math.sin(rad(rotation)),
    y: inputVector.x * Math.sin(rad(rotation)) + inputVector.y * Math.cos(rad(rotation))
  };

  // Target vector position
  const targetVector = {
    x: length * Math.cos(rad(targetAngle)),
    y: length * Math.sin(rad(targetAngle))
  };

  const alignment = Math.max(0, 100 - Math.abs(targetAngle - rotation));
  const isAligned = Math.abs(targetAngle - rotation) < 5;

  useEffect(() => {
    onUpdateContext({
      gameMode: GameMode.PRISM,
      currentStats: { angle: rotation, targetAngle, alignment: alignment.toFixed(1) }
    });
  }, [rotation, targetAngle, alignment, onUpdateContext]);

  const randomize = () => {
    setTargetAngle(Math.floor(Math.random() * 360));
  };

  return (
    <div className="h-full flex flex-col p-6 relative">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-purple-900/20 to-dark-bg -z-10" />

      <header className="mb-6 flex justify-between items-center border-b border-purple-500/30 pb-4">
        <div>
          <h2 className="text-2xl font-serif font-bold text-neon-pink flex items-center gap-2">
            <Sparkles className="text-purple-400" />
            PRISM OF THE ANCIENTS
          </h2>
          <p className="text-slate-400 text-sm mt-1">Rotate the crystal matrix to align the light beam with the rune.</p>
        </div>
        <button onClick={randomize} className="bg-slate-800 hover:bg-slate-700 p-2 rounded border border-white/10 text-white flex items-center gap-2 text-sm font-mono">
          <RefreshCw size={14} /> NEW RUNE
        </button>
      </header>

      <div className="flex-1 flex flex-col items-center justify-center gap-8">
        
        <div className="relative w-[400px] h-[400px] bg-black/40 rounded-full border border-purple-500/20 shadow-[0_0_50px_rgba(168,85,247,0.2)]">
           {/* Coordinate System Visualizer */}
           <svg viewBox="-200 -200 400 400" className="w-full h-full overflow-visible">
             <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
                </marker>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
             </defs>

             {/* Grid */}
             <line x1="-200" y1="0" x2="200" y2="0" stroke="#334155" strokeWidth="1" />
             <line x1="0" y1="-200" x2="0" y2="200" stroke="#334155" strokeWidth="1" />

             {/* Target Rune */}
             <line 
               x1="0" y1="0" 
               x2={targetVector.x} y2={targetVector.y} 
               stroke="rgba(255,255,255,0.1)" strokeWidth="2" strokeDasharray="5,5"
             />
             <circle 
               cx={targetVector.x} cy={targetVector.y} r="15" 
               className={`${isAligned ? 'fill-neon-pink' : 'fill-transparent stroke-neon-pink'} transition-all duration-500`}
             />
             
             {/* The Beam (Result Vector) */}
             <line 
               x1="0" y1="0" 
               x2={outputVector.x} y2={outputVector.y} 
               stroke={isAligned ? '#ff00ff' : '#00f3ff'} 
               strokeWidth="4" 
               filter="url(#glow)"
             />
             
             {/* The Crystal (Matrix Visualizer) */}
             <g transform={`rotate(${rotation})`}>
                <polygon points="-20,-30 20,-30 30,0 20,30 -20,30 -30,0" className="fill-purple-900/50 stroke-purple-400 stroke-1" />
                <circle cx="0" cy="0" r="5" className="fill-white" />
             </g>

             {/* Angle Arc */}
             <path d={`M 30 0 A 30 30 0 ${Math.abs(rotation) > 180 ? 1 : 0} ${rotation > 0 ? 1 : 0} ${30 * Math.cos(rad(rotation))} ${30 * Math.sin(rad(rotation))}`} fill="none" stroke="white" strokeOpacity="0.5" />
           </svg>
        </div>

        <div className="w-full max-w-md space-y-4 text-center">
          <div className="flex justify-between text-mono text-sm font-bold text-slate-400 uppercase">
            <span>Current: {rotation.toFixed(0)}°</span>
            <span>Loss: {Math.abs(targetAngle - rotation).toFixed(0)}</span>
            <span>Target: {targetAngle}°</span>
          </div>
          
          <input 
            type="range" min="-180" max="180" value={rotation}
            onChange={(e) => setRotation(Number(e.target.value))}
            className="w-full h-4 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-neon-pink hover:accent-purple-400 transition-all"
          />

          <div className="bg-slate-900 p-4 rounded font-mono text-xs text-slate-400 text-left border border-white/5">
            <div className="text-neon-pink mb-1">// MATRIX OPERATION</div>
            <p>x_new = x * cos(θ) - y * sin(θ)</p>
            <p>y_new = x * sin(θ) + y * cos(θ)</p>
            <p className="mt-2 text-white">
              Vector &lt;1, 0&gt; transformed by Matrix(θ={rotation}°) <br/>
              Result: &lt;{Math.cos(rad(rotation)).toFixed(2)}, {Math.sin(rad(rotation)).toFixed(2)}&gt;
            </p>
          </div>
        </div>

      </div>
    </div>
  );
};