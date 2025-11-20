import React, { useState, useEffect } from 'react';
import { Activity, Play, RefreshCw, ArrowDown } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, ReferenceDot, ReferenceLine } from 'recharts';
import { TutorContext, GameMode } from '../types';

interface DescentGameProps {
  onUpdateContext: (ctx: TutorContext) => void;
}

export const DescentGame: React.FC<DescentGameProps> = ({ onUpdateContext }) => {
  const [position, setPosition] = useState(-4.5); // Start far left
  const [learningRate, setLearningRate] = useState(0.1);
  const [steps, setSteps] = useState(0);
  
  // Loss function: y = x^2
  // Gradient: dy/dx = 2x
  const loss = (x: number) => x * x;
  const gradient = (x: number) => 2 * x;

  const dataPoints = [];
  for (let x = -5; x <= 5; x += 0.5) {
    dataPoints.push({ x, y: loss(x) });
  }

  useEffect(() => {
    onUpdateContext({
      gameMode: GameMode.DESCENT,
      currentStats: { 
        x: position.toFixed(2), 
        loss: loss(position).toFixed(2), 
        learningRate, 
        gradient: gradient(position).toFixed(2) 
      }
    });
  }, [position, learningRate, onUpdateContext]);

  const step = () => {
    const grad = gradient(position);
    const newPos = position - (learningRate * grad);
    setPosition(newPos);
    setSteps(s => s + 1);
  };

  const reset = () => {
    setPosition((Math.random() > 0.5 ? 1 : -1) * (3 + Math.random()));
    setSteps(0);
  };

  const isConverged = Math.abs(position) < 0.1;

  return (
    <div className="h-full flex flex-col p-6 relative">
      <div className="absolute inset-0 bg-[linear-gradient(to_bottom,_#000000_0%,_#062c18_100%)] -z-10" />
      
      {/* Grid Overlay Effect */}
      <div className="absolute inset-0 opacity-20 pointer-events-none" 
           style={{
             backgroundImage: `linear-gradient(rgba(0, 255, 0, 0.1) 1px, transparent 1px),
             linear-gradient(90deg, rgba(0, 255, 0, 0.1) 1px, transparent 1px)`,
             backgroundSize: '40px 40px',
             transform: 'perspective(500px) rotateX(60deg) translateY(-100px) scale(2)'
           }}>
      </div>

      <header className="mb-6 flex justify-between items-center border-b border-neon-green/30 pb-4 z-10">
        <div>
          <h2 className="text-2xl font-mono font-bold text-neon-green flex items-center gap-2">
            <Activity />
            CYBER-DEFENSE PROTOCOL
          </h2>
          <p className="text-slate-400 text-sm mt-1">Roll the packet down the Loss Landscape to minimize system error.</p>
        </div>
        <div className="text-right text-neon-green font-mono">
          <div className="text-xs opacity-70">CURRENT LOSS</div>
          <div className="text-3xl">{loss(position).toFixed(3)}</div>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 z-10">
        
        {/* Chart */}
        <div className="lg:col-span-3 bg-black/40 border border-neon-green/20 rounded-xl p-4 relative backdrop-blur-sm">
           <div className="absolute top-4 left-4 text-xs font-mono text-neon-green/60">LOSS MANIFOLD VISUALIZATION</div>
           <ResponsiveContainer width="100%" height="100%">
             <LineChart data={dataPoints} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
               <XAxis dataKey="x" type="number" domain={[-5, 5]} stroke="#334155" hide />
               <YAxis domain={[0, 25]} stroke="#334155" hide />
               <Line type="monotone" dataKey="y" stroke="#00ff00" strokeWidth={2} dot={false} />
               <ReferenceLine x={0} stroke="#1e293b" strokeDasharray="3 3" />
               {/* The Rolling Ball */}
               <ReferenceDot x={position} y={loss(position)} r={8} fill="#fff" stroke="#00ff00" strokeWidth={2} />
             </LineChart>
           </ResponsiveContainer>
           
           {isConverged && (
             <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm rounded-xl">
               <div className="text-center animate-pulse">
                 <h3 className="text-3xl font-mono font-bold text-neon-green">SYSTEM OPTIMIZED</h3>
                 <p className="text-white text-sm">Global Minimum Reached</p>
               </div>
             </div>
           )}
        </div>

        {/* Controls */}
        <div className="bg-slate-900/80 border border-neon-green/20 rounded-xl p-6 flex flex-col gap-6">
          <div>
            <label className="block text-xs font-bold text-neon-green mb-2 uppercase tracking-wider">
              Learning Rate (Step Size)
            </label>
            <input 
              type="range" min="0.01" max="1.2" step="0.01" value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              className="w-full accent-neon-green h-2 bg-slate-700 rounded appearance-none"
            />
            <div className="flex justify-between text-xs font-mono text-slate-400 mt-2">
              <span>Caution (0.01)</span>
              <span>{learningRate}</span>
              <span>Chaos (1.2)</span>
            </div>
          </div>

          <div className="bg-black p-4 rounded border border-neon-green/20 font-mono text-xs space-y-2 text-slate-300">
            <p>Current X: <span className="text-white">{position.toFixed(3)}</span></p>
            <p>Gradient (Slope): <span className={gradient(position) > 0 ? 'text-red-400' : 'text-blue-400'}>{gradient(position).toFixed(3)}</span></p>
            <p>Next Step: X - ({learningRate} * Slope)</p>
          </div>

          <button 
            onClick={step}
            disabled={isConverged}
            className="w-full py-4 bg-neon-green/10 border border-neon-green text-neon-green hover:bg-neon-green hover:text-black font-bold rounded transition-all flex items-center justify-center gap-2 disabled:opacity-50"
          >
            <Play size={16} /> RUN UPDATE STEP
          </button>
          
          <button 
            onClick={reset}
            className="w-full py-2 bg-slate-800 text-slate-400 hover:text-white rounded transition-colors text-sm flex items-center justify-center gap-2"
          >
            <RefreshCw size={14} /> RESET SYSTEM
          </button>

          <div className="mt-auto text-center">
            <span className="text-xs text-slate-500 uppercase tracking-widest">Epochs (Steps)</span>
            <div className="text-4xl font-mono text-white font-bold">{steps}</div>
          </div>
        </div>

      </div>
    </div>
  );
};