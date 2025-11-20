import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Network, Play, Pause, RotateCcw, Plus, Minus, Activity, CheckCircle, ZoomIn, ZoomOut, Move, Sliders, Eye } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TutorContext, GameMode } from '../types';

interface NetworkGameProps {
  onUpdateContext: (ctx: TutorContext) => void;
}

type ActivationKey = 'SIGMOID' | 'TANH' | 'RELU' | 'LEAKY_RELU' | 'ELU';

interface ActivationDef {
  name: string;
  func: (x: number) => number;
  deriv: (y: number) => number;
  color: string;
}

const ACTIVATION_FUNCTIONS: Record<ActivationKey, ActivationDef> = {
  SIGMOID: { name: 'Sigmoid', func: (x) => 1 / (1 + Math.exp(-x)), deriv: (y) => y * (1 - y), color: '#00f3ff' },
  TANH: { name: 'Tanh', func: (x) => Math.tanh(x), deriv: (y) => 1 - (y * y), color: '#d946ef' },
  RELU: { name: 'ReLU', func: (x) => Math.max(0, x), deriv: (y) => y > 0 ? 1 : 0.05, color: '#22c55e' },
  LEAKY_RELU: { name: 'L-ReLU', func: (x) => Math.max(0.01 * x, x), deriv: (y) => y > 0 ? 1 : 0.01, color: '#eab308' },
  ELU: { name: 'ELU', func: (x) => x >= 0 ? x : 1.0 * (Math.exp(x) - 1), deriv: (y) => y > 0 ? 1 : y + 1.0, color: '#f97316' }
};

const ACTIVATION_KEYS = Object.keys(ACTIVATION_FUNCTIONS) as ActivationKey[];

class SimpleNetwork {
  layerSizes: number[];
  activations: ActivationKey[];
  weights: number[][][];
  weightDeltas: number[][][];
  biases: number[][];
  values: number[][];
  preActivations: number[][];

  constructor(layerSizes: number[], activations: ActivationKey[]) {
    this.layerSizes = layerSizes;
    this.activations = activations;
    this.weights = [];
    this.weightDeltas = [];
    this.biases = [];
    this.values = [];
    this.preActivations = [];

    for (let i = 0; i < layerSizes.length; i++) {
      this.values.push(new Array(layerSizes[i]).fill(0));
      this.preActivations.push(new Array(layerSizes[i]).fill(0));
      
      if (i > 0) {
        const prevSize = layerSizes[i - 1];
        const currSize = layerSizes[i];
        const scale = 1 / Math.sqrt(prevSize);
        const layerWeights: number[][] = [];
        const layerDeltas: number[][] = [];
        for (let j = 0; j < prevSize; j++) {
          layerWeights.push(new Array(currSize).fill(0).map(() => ((Math.random() * 2) - 1) * scale));
          layerDeltas.push(new Array(currSize).fill(0));
        }
        this.weights.push(layerWeights);
        this.weightDeltas.push(layerDeltas);
        this.biases.push(new Array(currSize).fill(0.1));
      }
    }
  }

  setActivations(newActivations: ActivationKey[]) { this.activations = newActivations; }
  setLayerBias(layerIdx: number, value: number) {
    const biasIdx = layerIdx - 1;
    if (biasIdx >= 0 && biasIdx < this.biases.length) {
      this.biases[biasIdx].fill(value);
    }
  }

  forward(inputs: number[]) {
    this.values[0] = inputs;
    this.preActivations[0] = inputs;
    for (let i = 0; i < this.weights.length; i++) {
      const layerIndex = i + 1;
      const prevValues = this.values[i];
      const currentWeights = this.weights[i];
      const currentBiases = this.biases[i];
      const nextValues = [];
      const nextPreActivations = [];
      const activate = ACTIVATION_FUNCTIONS[this.activations[layerIndex]].func;

      for (let j = 0; j < this.layerSizes[layerIndex]; j++) {
        let sum = 0;
        for (let k = 0; k < prevValues.length; k++) sum += prevValues[k] * currentWeights[k][j];
        sum += currentBiases[j];
        if (!isFinite(sum)) sum = 0;
        nextPreActivations.push(sum);
        nextValues.push(activate(sum));
      }
      this.values[layerIndex] = nextValues;
      this.preActivations[layerIndex] = nextPreActivations;
    }
    return this.values[this.values.length - 1];
  }

  train(inputs: number[], targets: number[], learningRate: number) {
    this.forward(inputs);
    const layerErrors: number[][] = [];
    const outputLayerIdx = this.layerSizes.length - 1;
    const outputs = this.values[outputLayerIdx];
    const outputErrors = [];
    const outputDeriv = ACTIVATION_FUNCTIONS[this.activations[outputLayerIdx]].deriv;

    for(let i=0; i<outputs.length; i++) {
      outputErrors.push((targets[i] - outputs[i]) * outputDeriv(outputs[i]));
    }
    layerErrors[outputLayerIdx] = outputErrors;

    for (let i = this.weights.length - 1; i >= 0; i--) {
      const nextLayerErrors = layerErrors[i + 1];
      const currentLayerErrors = [];
      const currentValues = this.values[i];
      const currentLayerDeriv = i > 0 ? ACTIVATION_FUNCTIONS[this.activations[i]].deriv : () => 1;

      for (let k = 0; k < this.layerSizes[i]; k++) {
        for (let j = 0; j < this.layerSizes[i + 1]; j++) {
           const gradient = nextLayerErrors[j] * currentValues[k];
           if (!isFinite(gradient)) continue;
           const delta = Math.max(-1, Math.min(1, learningRate * gradient));
           this.weights[i][k][j] += delta;
           this.weightDeltas[i][k][j] = delta; 
        }
      }
      for (let j = 0; j < this.layerSizes[i + 1]; j++) {
        this.biases[i][j] += Math.max(-1, Math.min(1, learningRate * nextLayerErrors[j]));
      }
      if (i > 0) {
         for (let k = 0; k < this.layerSizes[i]; k++) {
            let sum = 0;
            for (let j = 0; j < this.layerSizes[i+1]; j++) sum += nextLayerErrors[j] * this.weights[i][k][j];
            if(!isFinite(sum)) sum = 0;
            currentLayerErrors.push(sum * currentLayerDeriv(currentValues[k]));
         }
         layerErrors[i] = currentLayerErrors;
      }
    }
  }
}

interface LayerDim {
    rows: number;
    cols: number;
}

export const NetworkGame: React.FC<NetworkGameProps> = ({ onUpdateContext }) => {
  // 3D Matrix State
  const [layerDims, setLayerDims] = useState<LayerDim[]>([
      { rows: 2, cols: 1 }, 
      { rows: 2, cols: 2 }, 
      { rows: 3, cols: 1 }, 
      { rows: 1, cols: 1 }
  ]);
  const [layerActivations, setLayerActivations] = useState<ActivationKey[]>(['LEAKY_RELU', 'LEAKY_RELU', 'LEAKY_RELU', 'SIGMOID']); 
  
  const [inputs, setInputs] = useState<number[]>([0, 1]);
  const [target, setTarget] = useState<number[]>([1]);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [epochs, setEpochs] = useState(0);
  const epochRef = useRef(0);
  const [learningRate, setLearningRate] = useState(0.1);
  const [simSpeed, setSimSpeed] = useState(200);
  const [isOptimized, setIsOptimized] = useState(false);
  const [lossHistory, setLossHistory] = useState<{epoch: number, loss: number}[]>([]);
  
  // 3D Viewport State
  const [rotation, setRotation] = useState(25); // Degrees
  const [viewScale, setViewScale] = useState(1);
  const [viewOffset, setViewOffset] = useState({ x: 0, y: 0 });
  const [viewSettings, setViewSettings] = useState({
      nodeSize: 12,
      textSize: 10,
      animWidth: 4
  });

  const isDraggingRef = useRef(false);
  const lastMouseRef = useRef({ x: 0, y: 0 });
  const svgRef = useRef<SVGSVGElement>(null);
  
  // Animation State
  const [direction, setDirection] = useState<'NONE' | 'FORWARD' | 'BACKWARD' | 'UPDATING'>('NONE');
  const [activeLayer, setActiveLayer] = useState(-1);

  // Network Refs & Visual Cache
  const networkRef = useRef<SimpleNetwork | null>(null);
  const [displayStats, setDisplayStats] = useState({
      values: [] as number[][],
      weights: [] as number[][][],
      deltas: [] as number[][][],
      biases: [] as number[][],
      loss: 0
  });

  // Initialize/Reset Network
  useEffect(() => {
      const layerSizes = layerDims.map(d => d.rows * d.cols);
      const net = new SimpleNetwork(layerSizes, layerActivations);
      
      // Sync inputs/targets size
      const inputSize = layerSizes[0];
      const outputSize = layerSizes[layerSizes.length - 1];
      
      if (inputs.length !== inputSize) setInputs(new Array(inputSize).fill(0));
      if (target.length !== outputSize) setTarget(new Array(outputSize).fill(0));
      
      networkRef.current = net;
      resetState();
      updateVisuals();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(layerDims)]);

  const resetState = () => {
      setEpochs(0);
      epochRef.current = 0;
      setLossHistory([]);
      setIsPlaying(false);
      setIsOptimized(false);
      setDirection('NONE');
      setActiveLayer(-1);
  };

  const updateVisuals = () => {
      if (!networkRef.current) return 0;
      const net = networkRef.current;
      const output = net.values[net.values.length - 1];
      let totalLoss = 0;
      for(let i=0; i<output.length; i++) totalLoss += 0.5 * Math.pow((target[i] || 0) - output[i], 2);
      
      setDisplayStats({
          values: JSON.parse(JSON.stringify(net.values)),
          weights: JSON.parse(JSON.stringify(net.weights)),
          deltas: JSON.parse(JSON.stringify(net.weightDeltas)),
          biases: JSON.parse(JSON.stringify(net.biases)),
          loss: totalLoss
      });
      return totalLoss;
  };

  // Trigger Forward on Input Change
  useEffect(() => {
      if (networkRef.current) {
          networkRef.current.forward(inputs);
          updateVisuals();
          setIsOptimized(false);
      }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputs, target, layerActivations]);

  // Sync Context for Tutor
  useEffect(() => {
      onUpdateContext({
          gameMode: GameMode.ARCHITECT,
          currentStats: {
              structure: layerDims.map(d => `${d.rows}x${d.cols}`).join('-'),
              epochs,
              loss: displayStats.loss.toFixed(5),
              lr: learningRate,
              optimized: isOptimized
          }
      });
  }, [displayStats.loss, epochs, isOptimized, onUpdateContext, layerDims, learningRate]);

  // Simulation Loop
  useEffect(() => {
    let animationFrame: number;
    let timer: NodeJS.Timeout;
    const CONVERGENCE = 0.00005;

    if (isPlaying && networkRef.current) {
        if (simSpeed < 1) { // Turbo
            const loop = () => {
                for(let k=0; k<10; k++) networkRef.current!.train(inputs, target, learningRate);
                epochRef.current += 10;
                setEpochs(epochRef.current);
                const loss = updateVisuals();
                
                setLossHistory(prev => {
                    const h = [...prev, { epoch: epochRef.current, loss }];
                    return h.length > 200 ? h.slice(-200) : h;
                });

                if (loss < CONVERGENCE) { setIsPlaying(false); setIsOptimized(true); }
                else if (isPlaying) animationFrame = requestAnimationFrame(loop);
            };
            loop();
        } else { // Animation
             timer = setTimeout(() => {
                if (direction === 'NONE' || direction === 'FORWARD') {
                    const next = activeLayer + 1;
                    if (activeLayer === -1) { networkRef.current!.forward(inputs); updateVisuals(); }
                    if (next < layerDims.length) { setDirection('FORWARD'); setActiveLayer(next); }
                    else { setDirection('BACKWARD'); setActiveLayer(layerDims.length - 1); }
                } else if (direction === 'BACKWARD') {
                    const prev = activeLayer - 1;
                    if (prev >= 0) setActiveLayer(prev);
                    else {
                        networkRef.current!.train(inputs, target, learningRate);
                        epochRef.current++;
                        setEpochs(epochRef.current);
                        const loss = updateVisuals();
                        setLossHistory(prev => [...prev.slice(-199), { epoch: epochRef.current, loss }]);
                        if (loss < CONVERGENCE) { setIsPlaying(false); setIsOptimized(true); setDirection('NONE'); setActiveLayer(-1); }
                        else { setDirection('UPDATING'); setActiveLayer(-1); }
                    }
                } else if (direction === 'UPDATING') {
                    setDirection('FORWARD'); setActiveLayer(0);
                }
             }, simSpeed);
        }
    }
    return () => { cancelAnimationFrame(animationFrame); clearTimeout(timer); };
  }, [isPlaying, simSpeed, direction, activeLayer, inputs, target, learningRate]);

  // --- 3D Projection Engine ---
  const project3D = (layerIdx: number, row: number, col: number) => {
      // Space Settings
      const layerSpacing = 250;
      const rowSpacing = 80;
      const colSpacing = 80;
      
      // Center the network
      const totalLayersWidth = (layerDims.length - 1) * layerSpacing;
      const xStart = -totalLayersWidth / 2;

      const dims = layerDims[layerIdx];
      const yStart = -((dims.rows - 1) * rowSpacing) / 2;
      const zStart = -((dims.cols - 1) * colSpacing) / 2;

      const x3d = xStart + (layerIdx * layerSpacing);
      const y3d = yStart + (row * rowSpacing);
      const z3d = zStart + (col * colSpacing);

      // Rotation
      const rad = (rotation * Math.PI) / 180;
      const cos = Math.cos(rad);
      const sin = Math.sin(rad);
      
      const xRot = x3d * cos - z3d * sin;
      const zRot = x3d * sin + z3d * cos;

      // Perspective
      const fov = 1000;
      const baseScale = fov / (fov - zRot); // Z gets smaller as it goes back
      
      // Apply Global Zoom (viewScale) to the perspective
      const finalScale = baseScale * viewScale;

      return {
          x: 400 + (xRot * finalScale) + viewOffset.x, // Apply zoom to position spread
          y: 300 + (y3d * finalScale) + viewOffset.y, // Apply zoom to position spread
          scale: finalScale, // Used for element sizing
          depth: zRot
      };
  };

  // --- Render List Generation ---
  const renderItems = useMemo(() => {
      const items: any[] = [];
      if (!displayStats.values.length) return items;

      // Generate Nodes
      layerDims.forEach((dim, lIdx) => {
          for(let r=0; r<dim.rows; r++) {
              for(let c=0; c<dim.cols; c++) {
                  const flatIdx = (r * dim.cols) + c;
                  const pos = project3D(lIdx, r, c);
                  const value = displayStats.values[lIdx]?.[flatIdx] || 0;
                  const bias = lIdx > 0 ? (displayStats.biases[lIdx-1]?.[flatIdx] || 0) : 0;
                  const isInput = lIdx === 0;
                  const isOutput = lIdx === layerDims.length - 1;
                  
                  items.push({
                      type: 'NODE',
                      key: `node-${lIdx}-${flatIdx}`,
                      lIdx, flatIdx, isInput, isOutput,
                      pos, value, bias,
                      depth: pos.depth // For sorting
                  });

                  // Controls (Attached to bottom of layer column)
                  if (r === dim.rows - 1 && c === Math.floor(dim.cols/2)) {
                      items.push({
                          type: 'LAYER_CONTROL',
                          key: `ctrl-${lIdx}`,
                          lIdx,
                          pos: { ...pos, y: pos.y + 60 * pos.scale },
                          depth: pos.depth + 10 // Draw in front
                      });
                  }
              }
          }
      });

      // Generate Connections
      displayStats.weights.forEach((layerWeights, lIdx) => {
          const fromDim = layerDims[lIdx];
          const toDim = layerDims[lIdx+1];
          if(!toDim) return;

          for(let r1=0; r1<fromDim.rows; r1++) {
              for(let c1=0; c1<fromDim.cols; c1++) {
                  const fromFlat = (r1 * fromDim.cols) + c1;
                  const fromPos = project3D(lIdx, r1, c1);
                  
                  for(let r2=0; r2<toDim.rows; r2++) {
                      for(let c2=0; c2<toDim.cols; c2++) {
                          const toFlat = (r2 * toDim.cols) + c2;
                          const toPos = project3D(lIdx+1, r2, c2);
                          const w = layerWeights[fromFlat]?.[toFlat] || 0;
                          const delta = displayStats.deltas[lIdx]?.[fromFlat]?.[toFlat] || 0;
                          
                          // Avg depth
                          const depth = (fromPos.depth + toPos.depth) / 2;
                          
                          items.push({
                              type: 'LINK',
                              key: `link-${lIdx}-${fromFlat}-${toFlat}`,
                              lIdx, fromFlat, toFlat,
                              fromPos, toPos, w, delta,
                              depth
                          });
                      }
                  }
              }
          }
      });

      // Sort Back-to-Front (Painter's Algorithm)
      return items.sort((a, b) => a.depth - b.depth);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layerDims, rotation, viewScale, viewOffset, displayStats, direction, activeLayer, inputs, target, viewSettings]);

  // UI Handlers
  const modifyLayer = (lIdx: number, dKey: 'rows' | 'cols', delta: number) => {
      setLayerDims(prev => {
          const next = [...prev];
          const newVal = Math.max(1, Math.min(8, next[lIdx][dKey] + delta));
          next[lIdx] = { ...next[lIdx], [dKey]: newVal };
          return next;
      });
      setIsOptimized(false);
  };

  const addLayer = () => {
      if(layerDims.length < 10) {
          setLayerDims([...layerDims.slice(0, -1), { rows: 2, cols: 2 }, layerDims[layerDims.length-1]]);
          setLayerActivations([...layerActivations.slice(0, -1), 'LEAKY_RELU', layerActivations[layerActivations.length-1]]);
          setIsOptimized(false);
      }
  };

  const removeLayer = () => {
      if(layerDims.length > 2) {
          const newDims = [...layerDims];
          newDims.splice(newDims.length - 2, 1);
          setLayerDims(newDims);
          const newActs = [...layerActivations];
          newActs.splice(newActs.length - 2, 1);
          setLayerActivations(newActs);
          setIsOptimized(false);
      }
  };

  // Mouse Controls
  const handleMouseDown = (e: React.MouseEvent) => { 
      isDraggingRef.current = true; 
      lastMouseRef.current = { x: e.clientX, y: e.clientY }; 
  };

  const handleMouseMove = (e: React.MouseEvent) => {
      if (!isDraggingRef.current) return;
      
      const dx = e.clientX - lastMouseRef.current.x;
      const dy = e.clientY - lastMouseRef.current.y;
      
      if (e.buttons === 1) {
          // Left Click: Rotate
          setRotation(r => r + dx * 0.5);
      } else if (e.buttons === 2 || e.buttons === 3) {
          // Right Click or Both: Pan
          setViewOffset(prev => ({
              x: prev.x + dx,
              y: prev.y + dy
          }));
      }
      
      lastMouseRef.current = { x: e.clientX, y: e.clientY };
  };
  const handleMouseUp = () => { isDraggingRef.current = false; };
  const handleWheel = (e: React.WheelEvent) => { setViewScale(s => Math.max(0.05, Math.min(5, s - e.deltaY * 0.001))); };

  return (
    <div className="h-full flex flex-col p-6 relative select-none">
      <div className="absolute inset-0 bg-gradient-to-tr from-slate-900 via-blue-900/10 to-black -z-10" />
      
      <style>{`
        @keyframes flow { from { stroke-dashoffset: 20; } to { stroke-dashoffset: 0; } }
        .anim-flow { animation: flow ${Math.max(100, simSpeed)}ms linear infinite; }
      `}</style>

      {/* Top Bar */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 bg-panel-bg/90 border border-white/10 rounded-full p-2 flex items-center gap-4 backdrop-blur shadow-xl">
         <div className="px-4 border-r border-white/10 flex items-center gap-2 text-blue-400 font-bold font-mono text-sm">
            <Move size={18} /> 3D MATRIX VIEW
         </div>
         <div className="flex items-center gap-2">
             <button onClick={addLayer} className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-blue-600 rounded-full text-xs font-bold border border-white/5"><Plus size={14}/> LAYER</button>
             <button onClick={removeLayer} className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-red-600 rounded-full text-xs font-bold border border-white/5"><Minus size={14}/> LAYER</button>
         </div>
         <div className="px-4 text-xs font-mono">
             <span className="text-slate-500">EPOCH </span>
             <span className="text-white font-bold">{epochs}</span>
         </div>
      </div>

      {/* Loss Graph */}
      <div className="absolute top-6 left-6 z-20 w-64 h-32 bg-panel-bg/80 backdrop-blur border border-white/10 rounded-xl p-2 shadow-xl">
         <div className="flex justify-between items-center px-2 mb-1">
            <span className="text-[10px] font-bold text-slate-400">LOSS</span>
            <span className={`text-[10px] font-mono font-bold ${displayStats.loss < 0.01 ? 'text-green-400' : 'text-red-400'}`}>{displayStats.loss.toFixed(6)}</span>
         </div>
         <ResponsiveContainer width="100%" height="80%">
            <AreaChart data={lossHistory}>
                 <defs>
                    <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00f3ff" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#00f3ff" stopOpacity={0}/>
                    </linearGradient>
                </defs>
                <Area type="monotone" dataKey="loss" stroke="#00f3ff" strokeWidth={2} fill="url(#colorLoss)" isAnimationActive={false} />
            </AreaChart>
         </ResponsiveContainer>
      </div>

      {/* VIEW SETTINGS PANEL (Top Right) */}
      <div className="absolute top-20 right-6 z-30 bg-panel-bg/80 backdrop-blur border border-white/10 rounded-xl p-4 shadow-xl w-48 flex flex-col gap-4" onMouseDown={(e) => e.stopPropagation()}>
          <div className="flex items-center gap-2 text-slate-400 text-xs font-bold border-b border-white/10 pb-2">
              <Eye size={14} /> VIEW SETTINGS
          </div>
          
          <div>
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">NODE SIZE</div>
              <input 
                  type="range" min="5" max="30" step="1" 
                  value={viewSettings.nodeSize} 
                  onChange={(e) => setViewSettings(s => ({...s, nodeSize: Number(e.target.value)}))}
                  className="w-full accent-blue-500 h-1 bg-slate-700 rounded"
              />
          </div>

          <div>
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">TEXT SIZE</div>
              <input 
                  type="range" min="0" max="20" step="1" 
                  value={viewSettings.textSize} 
                  onChange={(e) => setViewSettings(s => ({...s, textSize: Number(e.target.value)}))}
                  className="w-full accent-blue-500 h-1 bg-slate-700 rounded"
              />
          </div>

          <div>
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">ANIM WIDTH</div>
              <input 
                  type="range" min="1" max="15" step="1" 
                  value={viewSettings.animWidth} 
                  onChange={(e) => setViewSettings(s => ({...s, animWidth: Number(e.target.value)}))}
                  className="w-full accent-green-500 h-1 bg-slate-700 rounded"
              />
          </div>
      </div>

      {/* Bottom Controls */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-30 bg-panel-bg/90 border border-white/10 rounded-2xl p-4 flex items-center gap-6 backdrop-blur shadow-2xl">
         <button onClick={() => setIsPlaying(!isPlaying)} className={`w-12 h-12 rounded-xl flex items-center justify-center ${isPlaying ? 'bg-red-500 text-white' : 'bg-blue-500 text-white'}`}>
            {isPlaying ? <Pause size={24} fill="currentColor"/> : <Play size={24} fill="currentColor"/>}
         </button>
         <button onClick={resetState} className="w-12 h-12 bg-slate-800 text-slate-400 hover:text-white rounded-xl flex items-center justify-center"><RotateCcw size={20}/></button>
         <div className="w-px h-10 bg-white/10" />
         <div className="flex flex-col w-48 gap-1">
            <div className="flex justify-between text-[10px] font-bold text-slate-500 uppercase">
                <span>Speed</span>
                <span className={simSpeed < 1 ? 'text-neon-green' : ''}>{simSpeed < 1 ? 'TURBO' : `${simSpeed.toFixed(1)}ms`}</span>
            </div>
            <input type="range" min="0.1" max="200" step="0.1" value={200.1 - simSpeed} onChange={(e) => setSimSpeed(200.1 - Number(e.target.value))} className="w-full accent-blue-500 h-1.5 bg-slate-700 rounded appearance-none"/>
         </div>
      </div>
      
      {/* Visualization Canvas */}
      <div className="flex-1 bg-slate-950 rounded-xl border border-white/10 relative shadow-inner overflow-hidden">
         {isOptimized && (
             <div className="absolute top-10 left-1/2 -translate-x-1/2 z-20 bg-black/80 backdrop-blur border border-neon-green px-6 py-3 rounded-full flex items-center gap-3 animate-bounce">
                 <CheckCircle className="text-neon-green" /> <span className="text-neon-green font-bold font-mono">OPTIMIZED</span>
             </div>
         )}
         <svg 
            ref={svgRef}
            className="w-full h-full cursor-move"
            onMouseDown={handleMouseDown} 
            onMouseMove={handleMouseMove} 
            onMouseUp={handleMouseUp} 
            onMouseLeave={handleMouseUp} 
            onWheel={handleWheel}
            onContextMenu={(e) => e.preventDefault()}
         >
             <defs>
                <filter id="glow-green"><feGaussianBlur stdDeviation="4"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                <filter id="glow-red"><feGaussianBlur stdDeviation="4"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                <filter id="glow-line"><feGaussianBlur stdDeviation="2"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>
             </defs>
             {renderItems.map((item) => {
                 if (item.type === 'LINK') {
                     const { key, fromPos, toPos, w, delta, lIdx } = item;
                     const opacity = Math.min(1, Math.abs(w));
                     const isUpdating = direction === 'UPDATING' && Math.abs(delta) > 0.001;
                     const isActive = (direction === 'FORWARD' && activeLayer === lIdx) || (direction === 'BACKWARD' && activeLayer === lIdx+1);
                     const color = w > 0 ? '#00f3ff' : '#ff0055';
                     
                     const strokeColor = isActive ? (direction==='FORWARD'?'#00ff00':'#ff0000') : isUpdating ? '#ffff00' : color;
                     const strokeWidth = (isActive || isUpdating ? viewSettings.animWidth : Math.max(0.5, Math.abs(w)*3)) * item.fromPos.scale;

                     return (
                         <line 
                            key={key} x1={fromPos.x} y1={fromPos.y} x2={toPos.x} y2={toPos.y} 
                            stroke={strokeColor}
                            strokeWidth={strokeWidth}
                            strokeOpacity={isActive || isUpdating ? 0.9 : opacity * 0.3}
                            strokeDasharray={isActive || isUpdating ? "10, 10" : ""}
                            className={isActive || isUpdating ? "anim-flow" : ""}
                            filter={isActive || isUpdating ? "url(#glow-line)" : ""}
                         />
                     );
                 } else if (item.type === 'NODE') {
                     const { key, pos, value, isInput, isOutput, lIdx, flatIdx, bias } = item;
                     const isActive = isActiveLayer(lIdx);
                     const baseR = isActive ? viewSettings.nodeSize * 1.3 : viewSettings.nodeSize;
                     const r = baseR * pos.scale;

                     const fillOpacity = isInput ? 1 : 0.2 + (value * 0.8);
                     
                     return (
                         <g key={key}>
                             <circle cx={pos.x} cy={pos.y} r={r} 
                                fill={isActive ? (direction==='FORWARD'?'#00ff00':'#ff0000') : '#00f3ff'} 
                                fillOpacity={Math.min(1, Math.max(0, fillOpacity))} 
                                stroke={isActive ? (direction==='FORWARD'?'#00ff00':'#ff0000') : '#00f3ff'}
                                strokeWidth={2 * pos.scale}
                                filter={isActive ? `url(#glow-${direction==='FORWARD'?'green':'red'})` : ''}
                             />
                             {pos.scale > 0.4 && viewSettings.textSize > 0 && (
                                 <text x={pos.x} y={pos.y + (r * 1.5) + 5} textAnchor="middle" fill="white" fontSize={viewSettings.textSize * pos.scale} className="font-mono font-bold pointer-events-none">
                                     {value.toFixed(2)}
                                 </text>
                             )}
                             
                             {/* Interactive Inputs/Targets */}
                             {(isInput || isOutput) && (
                                 <foreignObject x={pos.x - 40} y={pos.y - 25} width={80} height={50} style={{transform: `scale(${pos.scale})`, transformOrigin: `${pos.x}px ${pos.y}px`}}>
                                     <div className="flex flex-col items-center" onMouseDown={(e) => e.stopPropagation()}>
                                         <input 
                                            type="range" min="0" max="1" step="0.1" 
                                            value={isInput ? (inputs[flatIdx]||0) : (target[flatIdx]||0)}
                                            onChange={(e) => {
                                                const val = Number(e.target.value);
                                                if (isInput) { const n=[...inputs]; n[flatIdx]=val; setInputs(n); }
                                                else { const n=[...target]; n[flatIdx]=val; setTarget(n); }
                                            }}
                                            className={`w-16 h-1 rounded appearance-none ${isInput ? 'bg-blue-600 accent-blue-400' : 'bg-amber-600 accent-amber-400'}`}
                                         />
                                     </div>
                                 </foreignObject>
                             )}

                             {/* Bias Slider (Hidden Layers) */}
                             {!isInput && !isOutput && pos.scale > 0.5 && (
                                  <foreignObject x={pos.x - 30} y={pos.y + 35*pos.scale} width={60} height={30} style={{transform: `scale(${pos.scale})`, transformOrigin: `${pos.x}px ${pos.y}px`}}>
                                      <div className="flex justify-center" onMouseDown={(e) => e.stopPropagation()}>
                                          <input 
                                            type="range" min="-1" max="1" step="0.1" value={bias || 0}
                                            onChange={(e) => networkRef.current?.setLayerBias(lIdx, Number(e.target.value))}
                                            className="w-10 h-1 bg-purple-900 accent-purple-500 rounded appearance-none opacity-50 hover:opacity-100"
                                          />
                                      </div>
                                  </foreignObject>
                             )}
                         </g>
                     );
                 } else if (item.type === 'LAYER_CONTROL') {
                     const { key, pos, lIdx } = item;
                     return (
                         <foreignObject key={key} x={pos.x - 50} y={pos.y} width={100} height={50} style={{transform: `scale(${pos.scale})`, transformOrigin: `${pos.x}px ${pos.y}px`}}>
                             <div className="flex flex-col gap-1 items-center bg-black/50 backdrop-blur rounded p-1 border border-white/10" onMouseDown={(e) => e.stopPropagation()}>
                                 <div className="flex gap-1">
                                     <span className="text-[8px] font-mono text-slate-400 w-4">ROW</span>
                                     <button onClick={() => modifyLayer(lIdx, 'rows', 1)} className="w-4 h-4 bg-slate-700 text-[8px] hover:bg-blue-600 rounded">+</button>
                                     <button onClick={() => modifyLayer(lIdx, 'rows', -1)} className="w-4 h-4 bg-slate-700 text-[8px] hover:bg-red-600 rounded">-</button>
                                 </div>
                                 <div className="flex gap-1">
                                     <span className="text-[8px] font-mono text-slate-400 w-4">COL</span>
                                     <button onClick={() => modifyLayer(lIdx, 'cols', 1)} className="w-4 h-4 bg-slate-700 text-[8px] hover:bg-blue-600 rounded">+</button>
                                     <button onClick={() => modifyLayer(lIdx, 'cols', -1)} className="w-4 h-4 bg-slate-700 text-[8px] hover:bg-red-600 rounded">-</button>
                                 </div>
                             </div>
                         </foreignObject>
                     );
                 }
                 return null;
             })}
         </svg>
      </div>
    </div>
  );

  function isActiveLayer(lIdx: number) {
      if (direction === 'FORWARD' && activeLayer === lIdx) return true;
      if (direction === 'BACKWARD' && activeLayer === lIdx) return true;
      return false;
  }
};