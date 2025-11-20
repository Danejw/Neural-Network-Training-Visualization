import React, { useState, useEffect, useRef } from 'react';
import { RefreshCw, Mountain, Wind, Move } from 'lucide-react';
import { TutorContext, GameMode } from '../types';

interface QuantumGameProps {
  onUpdateContext: (ctx: TutorContext) => void;
}

// Pure function for loss calculation to be used in search and render
const calculateLoss = (x: number, z: number, seed: any) => {
  const bowl = (x * x + z * z) * 0.15; // Main global shape
  const bumps = Math.cos(x * seed.freq + seed.phaseX) * seed.amp + 
                Math.cos(z * seed.freq + seed.phaseZ) * seed.amp;
  return bowl + bumps + 3; // +3 to keep it positive visually
};

export const QuantumGame: React.FC<QuantumGameProps> = ({ onUpdateContext }) => {
  // Simulation State
  const [ballPos, setBallPos] = useState({ x: 4, z: 4 });
  const [velocity, setVelocity] = useState({ x: 0, z: 0 });
  const [globalMin, setGlobalMin] = useState({ x: 0, z: 0 }); // The true target
  const [learningRate, setLearningRate] = useState(0.05);
  const [momentum, setMomentum] = useState(0.8);
  const [steps, setSteps] = useState(0);
  const [terrainSeed, setTerrainSeed] = useState({
    freq: 1.5,
    amp: 1.2,
    phaseX: Math.random() * Math.PI,
    phaseZ: Math.random() * Math.PI
  });
  const [isRunning, setIsRunning] = useState(false);
  
  // Rendering Ref
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const rotationRef = useRef(0.5); // Start at a slight angle

  // Mouse Interaction Refs
  const isDraggingRef = useRef(false);
  const lastMouseXRef = useRef(0);

  // Helper for current instance
  const getLoss = (x: number, z: number) => calculateLoss(x, z, terrainSeed);

  // Calculate True Global Minimum when map changes
  useEffect(() => {
    let minLoss = Infinity;
    let bestX = 0;
    let bestZ = 0;

    // 1. Coarse Search
    for (let x = -6; x <= 6; x += 0.2) {
      for (let z = -6; z <= 6; z += 0.2) {
        const l = calculateLoss(x, z, terrainSeed);
        if (l < minLoss) {
          minLoss = l;
          bestX = x;
          bestZ = z;
        }
      }
    }

    // 2. Fine Refinement around coarse minimum
    for (let x = bestX - 0.3; x <= bestX + 0.3; x += 0.02) {
      for (let z = bestZ - 0.3; z <= bestZ + 0.3; z += 0.02) {
        const l = calculateLoss(x, z, terrainSeed);
        if (l < minLoss) {
          minLoss = l;
          bestX = x;
          bestZ = z;
        }
      }
    }

    setGlobalMin({ x: bestX, z: bestZ });
  }, [terrainSeed]);

  // Calculate Gradient using finite differences
  const getGradient = (x: number, z: number) => {
    const h = 0.01;
    const loss = getLoss(x, z);
    const dx = (getLoss(x + h, z) - loss) / h;
    const dz = (getLoss(x, z + h) - loss) / h;
    return { dx, dz };
  };

  // Win condition: Distance to True Global Min
  const distToMin = Math.sqrt(Math.pow(ballPos.x - globalMin.x, 2) + Math.pow(ballPos.z - globalMin.z, 2));
  const isConverged = distToMin < 0.8;
  const isStuck = steps > 10 && Math.abs(velocity.x) < 0.001 && Math.abs(velocity.z) < 0.001 && !isConverged;

  // Main Update Loop
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      const { dx, dz } = getGradient(ballPos.x, ballPos.z);
      
      // Momentum Logic: v_new = (momentum * v_old) - (lr * gradient)
      const newVx = (velocity.x * momentum) - (learningRate * dx);
      const newVz = (velocity.z * momentum) - (learningRate * dz);
      
      const newX = ballPos.x + newVx;
      const newZ = ballPos.z + newVz;

      // Boundary checks
      if (Math.abs(newX) < 6) {
        setBallPos(prev => ({ ...prev, x: newX }));
        setVelocity(prev => ({ ...prev, x: newVx }));
      } else {
        setVelocity(prev => ({ ...prev, x: -prev.x * 0.5 })); // Bounce off walls
      }

      if (Math.abs(newZ) < 6) {
        setBallPos(prev => ({ ...prev, z: newZ }));
        setVelocity(prev => ({ ...prev, z: newVz }));
      } else {
        setVelocity(prev => ({ ...prev, z: -prev.z * 0.5 }));
      }

      setSteps(s => s + 1);

      // Stop if converged significantly to a spot (local or global)
      if (Math.abs(dx) < 0.001 && Math.abs(dz) < 0.001 && Math.abs(newVx) < 0.001 && Math.abs(newVz) < 0.001) {
        setIsRunning(false);
      }

    }, 30);

    return () => clearInterval(interval);
  }, [isRunning, ballPos, velocity, learningRate, momentum, terrainSeed]);

  // Context Update
  useEffect(() => {
    onUpdateContext({
      gameMode: GameMode.QUANTUM,
      currentStats: {
        x: ballPos.x.toFixed(2),
        z: ballPos.z.toFixed(2),
        loss: getLoss(ballPos.x, ballPos.z).toFixed(2),
        momentum,
        isStuck,
        distToGoal: distToMin.toFixed(2)
      }
    });
  }, [ballPos, momentum, isStuck, distToMin, onUpdateContext]);

  // Mouse Event Handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    isDraggingRef.current = true;
    lastMouseXRef.current = e.clientX;
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingRef.current) return;
    const delta = e.clientX - lastMouseXRef.current;
    rotationRef.current += delta * 0.01; // Rotation sensitivity
    lastMouseXRef.current = e.clientX;
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  // Rendering Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    const render = () => {
      ctx.fillStyle = '#0a0a0a';
      ctx.fillRect(0, 0, width, height);

      // Projection Parameters
      const scale = 30;
      const centerX = width / 2;
      const centerY = height / 2;
      
      const project = (x: number, y: number, z: number): {x: number, y: number, scale: number} => {
        // Rotate around Y axis
        const cos = Math.cos(rotationRef.current);
        const sin = Math.sin(rotationRef.current);
        const rotX = x * cos - z * sin;
        const rotZ = x * sin + z * cos;
        
        // Simple perspective
        const depth = 400 - rotZ * 10;
        const perspective = 400 / depth; 
        
        return {
          x: centerX + rotX * scale * 0.8, // Slight zoom out
          y: centerY + 100 - (y * scale * 0.8),
          scale: perspective
        };
      };

      // Draw Mesh
      ctx.strokeStyle = 'rgba(0, 255, 255, 0.15)';
      ctx.lineWidth = 1;

      const gridSize = 6;
      const step = 0.5;

      // Draw lines along X
      for (let z = -gridSize; z <= gridSize; z += step) {
        ctx.beginPath();
        let first = true;
        for (let x = -gridSize; x <= gridSize; x += 0.2) {
          const y = getLoss(x, z);
          const p = project(x, y, z);
          if (first) { ctx.moveTo(p.x, p.y); first = false; }
          else { ctx.lineTo(p.x, p.y); }
        }
        ctx.stroke();
      }

      // Draw lines along Z
      for (let x = -gridSize; x <= gridSize; x += step) {
        ctx.beginPath();
        let first = true;
        for (let z = -gridSize; z <= gridSize; z += 0.2) {
          const y = getLoss(x, z);
          const p = project(x, y, z);
          if (first) { ctx.moveTo(p.x, p.y); first = false; }
          else { ctx.lineTo(p.x, p.y); }
        }
        ctx.stroke();
      }

      // Draw Global Minima (Target)
      const minP = project(globalMin.x, getLoss(globalMin.x, globalMin.z), globalMin.z);
      ctx.beginPath();
      // Scale dot by perspective
      const minRadius = Math.max(2, 6 * (minP.scale || 1)); 
      ctx.arc(minP.x, minP.y, minRadius, 0, Math.PI * 2);
      ctx.fillStyle = '#00ff00';
      ctx.shadowColor = '#00ff00';
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Draw Ball
      const ballY = getLoss(ballPos.x, ballPos.z);
      const ballP = project(ballPos.x, ballY, ballPos.z);
      
      // Draw "Shadow"/Drop line
      const shadowP = project(ballPos.x, 0, ballPos.z);
      ctx.beginPath();
      ctx.moveTo(ballP.x, ballP.y);
      ctx.lineTo(shadowP.x, shadowP.y);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.setLineDash([2, 2]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Sphere
      const radius = Math.max(2, 8 * (ballP.scale || 1));
      
      ctx.beginPath();
      ctx.arc(ballP.x, ballP.y, radius, 0, Math.PI * 2);
      
      // Gradient for 3D look
      const grad = ctx.createRadialGradient(ballP.x - 2, ballP.y - 2, 0, ballP.x, ballP.y, radius);
      grad.addColorStop(0, '#ffffff');
      grad.addColorStop(1, isConverged ? '#00ff00' : isStuck ? '#ff0000' : '#0088ff');
      
      ctx.fillStyle = grad;
      ctx.fill();

      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animationRef.current);
  }, [ballPos, terrainSeed, globalMin, isConverged, isStuck]);

  const reset = () => {
    setBallPos({ 
      x: (Math.random() > 0.5 ? 1 : -1) * (3 + Math.random()), 
      z: (Math.random() > 0.5 ? 1 : -1) * (3 + Math.random()) 
    });
    setVelocity({ x: 0, z: 0 });
    setSteps(0);
    setIsRunning(false);
  };

  const newMap = () => {
    setTerrainSeed({
      freq: 1.0 + Math.random(),
      amp: 0.5 + Math.random(),
      phaseX: Math.random() * Math.PI,
      phaseZ: Math.random() * Math.PI
    });
    // Note: ball reset happens via reset() which is often called manually, 
    // but for new map we probably want to keep ball where it is or reset it.
    // Let's just stop running.
    setIsRunning(false);
    setSteps(0);
  };

  return (
    <div className="h-full flex flex-col p-6 relative">
      <div className="absolute inset-0 bg-gradient-to-b from-slate-900 via-purple-900/20 to-black -z-10" />
      
      <header className="mb-6 flex justify-between items-center border-b border-indigo-500/30 pb-4 z-10">
        <div>
          <h2 className="text-2xl font-mono font-bold text-indigo-400 flex items-center gap-2">
            <Mountain />
            THE QUANTUM VALLEY
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Navigate complex non-convex loss landscapes. Avoid local minima traps using momentum.
          </p>
        </div>
        <div className="text-right text-indigo-400 font-mono">
          <div className="text-xs opacity-70">CURRENT LOSS</div>
          <div className={`text-3xl ${isConverged ? 'text-neon-green' : isStuck ? 'text-red-500' : 'text-white'}`}>
            {getLoss(ballPos.x, ballPos.z).toFixed(3)}
          </div>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 z-10">
        
        {/* 3D Canvas */}
        <div className="lg:col-span-3 bg-black/50 border border-indigo-500/20 rounded-xl overflow-hidden relative">
           <canvas 
             ref={canvasRef} 
             width={800} 
             height={500}
             className="w-full h-full object-cover cursor-move"
             onMouseDown={handleMouseDown}
             onMouseMove={handleMouseMove}
             onMouseUp={handleMouseUp}
             onMouseLeave={handleMouseUp}
           />
           
           {/* Status Overlay */}
           <div className="absolute top-4 left-4 space-y-1 pointer-events-none">
             <div className="flex items-center gap-2 text-xs font-mono text-indigo-300">
               <span className="w-3 h-3 rounded-full bg-green-500 shadow-[0_0_10px_#00ff00]"></span> GLOBAL MINIMUM (GOAL)
             </div>
             <div className="flex items-center gap-2 text-xs font-mono text-indigo-300">
               <span className="w-3 h-3 rounded-full bg-blue-500"></span> YOUR AGENT
             </div>
             {isStuck && (
                <div className="mt-2 px-3 py-1 bg-red-500/20 border border-red-500 text-red-400 text-xs font-bold animate-pulse rounded">
                  ⚠️ STUCK IN LOCAL MINIMUM
                </div>
             )}
              {isConverged && (
                <div className="mt-2 px-3 py-1 bg-green-500/20 border border-green-500 text-green-400 text-xs font-bold animate-pulse rounded">
                  ✓ CONVERGED TO GLOBAL MINIMUM
                </div>
             )}
           </div>
           
           {/* Drag Hint */}
           <div className="absolute bottom-4 right-4 text-xs font-mono text-indigo-400/50 flex items-center gap-2 pointer-events-none select-none">
             <Move size={12} /> DRAG TO ROTATE VIEW
           </div>
        </div>

        {/* Controls */}
        <div className="bg-slate-900/80 border border-indigo-500/20 rounded-xl p-6 flex flex-col gap-6 backdrop-blur-md">
          
          {/* Learning Rate */}
          <div>
            <label className="block text-xs font-bold text-indigo-400 mb-2 uppercase tracking-wider flex justify-between">
              <span>Learning Rate</span>
              <span>{learningRate.toFixed(2)}</span>
            </label>
            <input 
              type="range" min="0.01" max="0.5" step="0.01" value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              className="w-full accent-indigo-500 h-2 bg-slate-700 rounded appearance-none"
            />
          </div>

          {/* Momentum */}
          <div>
            <label className="block text-xs font-bold text-pink-400 mb-2 uppercase tracking-wider flex justify-between">
              <span className="flex items-center gap-2"><Wind size={14}/> Momentum</span>
              <span>{momentum.toFixed(2)}</span>
            </label>
            <input 
              type="range" min="0.0" max="0.99" step="0.01" value={momentum}
              onChange={(e) => setMomentum(Number(e.target.value))}
              className="w-full accent-pink-500 h-2 bg-slate-700 rounded appearance-none"
            />
            <p className="text-[10px] text-slate-500 mt-1">
              Higher momentum helps roll over small hills (local minima).
            </p>
          </div>

          {/* Info Box */}
          <div className="bg-black p-4 rounded border border-white/10 font-mono text-xs space-y-2 text-slate-300">
            <p>Position: <span className="text-white">({ballPos.x.toFixed(1)}, {ballPos.z.toFixed(1)})</span></p>
            <p>Velocity: <span className="text-pink-400">{Math.sqrt(velocity.x**2 + velocity.z**2).toFixed(3)}</span></p>
            <p>Dist to Goal: <span className={distToMin < 2 ? 'text-green-400' : 'text-white'}>{distToMin.toFixed(2)}</span></p>
          </div>

          <div className="flex flex-col gap-2 mt-auto">
             <button 
              onClick={() => setIsRunning(!isRunning)}
              className={`w-full py-3 font-bold rounded transition-all flex items-center justify-center gap-2 border ${
                isRunning 
                ? 'bg-red-500/10 border-red-500 text-red-500 hover:bg-red-500 hover:text-white' 
                : 'bg-indigo-500/10 border-indigo-500 text-indigo-500 hover:bg-indigo-500 hover:text-white'
              }`}
            >
              {isRunning ? 'PAUSE SIMULATION' : 'START DESCENT'}
            </button>

            <div className="grid grid-cols-2 gap-2">
               <button 
                onClick={reset}
                className="py-2 bg-slate-800 text-slate-300 hover:text-white rounded transition-colors text-xs font-bold"
              >
                RESET BALL
              </button>
              <button 
                onClick={newMap}
                className="py-2 bg-slate-800 text-slate-300 hover:text-white rounded transition-colors text-xs font-bold flex items-center justify-center gap-1"
              >
                <RefreshCw size={12} /> NEW MAP
              </button>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
};