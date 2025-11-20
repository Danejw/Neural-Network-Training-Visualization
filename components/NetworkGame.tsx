import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Network, Play, Pause, RotateCcw, Plus, Minus, Activity, CheckCircle, ZoomIn, ZoomOut, Move, Sliders, Eye, GripHorizontal, BoxSelect } from 'lucide-react';
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
        
        // Biases for current layer
        // Initialize to small positive value to prevent "Dead ReLU" at start
        // Randomize slightly to ensure visual distinctness on reset (0.01 to 0.21)
        this.biases.push(new Array(currSize).fill(0).map(() => 0.01 + Math.random() * 0.2));
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
  
  // UI Panel Positions
  const [lossGraphPos, setLossGraphPos] = useState({ x: 24, y: 80 });
  const [viewSettingsPos, setViewSettingsPos] = useState({ x: 24, y: 230 });
  
  // 3D Viewport State
  const [cameraAngle, setCameraAngle] = useState({ h: 25, v: 15 }); // Horizontal (Yaw), Vertical (Pitch)
  const [viewScale, setViewScale] = useState(1);
  const [viewOffset, setViewOffset] = useState({ x: 0, y: 0 });
  const [isOrtho, setIsOrtho] = useState(false); // Orthographic vs Perspective
  
  const [viewSettings, setViewSettings] = useState({
      nodeSize: 12,
      textSize: 10,
      animWidth: 4,
      sliderSize: 1.0
  });

  // Interaction Refs
  const isDraggingCanvasRef = useRef(false);
  const draggingUIRef = useRef<{ id: 'LOSS' | 'SETTINGS', offsetX: number, offsetY: number } | null>(null);
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
      preActivations: [] as number[][],
      loss: 0
  });

  // Initialize Network (Structure Change)
  useEffect(() => {
      const layerSizes = layerDims.map(d => d.rows * d.cols);
      const net = new SimpleNetwork(layerSizes, layerActivations);
      
      // Sync inputs/targets size
      const inputSize = layerSizes[0];
      const outputSize = layerSizes[layerSizes.length - 1];
      
      if (inputs.length !== inputSize) setInputs(new Array(inputSize).fill(0));
      if (target.length !== outputSize) setTarget(new Array(outputSize).fill(0));
      
      networkRef.current = net;
      
      // Reset Counters manually
      setEpochs(0);
      epochRef.current = 0;
      setLossHistory([]);
      setIsPlaying(false);
      setIsOptimized(false);
      setDirection('NONE');
      setActiveLayer(-1);

      updateVisuals();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(layerDims)]);

  // Hard Reset (Randomize Weights)
  const resetSimulation = () => {
      const layerSizes = layerDims.map(d => d.rows * d.cols);
      // Create NEW network -> Randomizes weights and biases
      const net = new SimpleNetwork(layerSizes, layerActivations);
      networkRef.current = net;

      setEpochs(0);
      epochRef.current = 0;
      setLossHistory([]);
      setIsPlaying(false);
      setIsOptimized(false);
      setDirection('NONE');
      setActiveLayer(-1);
      
      // Initial Forward
      net.forward(inputs);
      updateVisuals();
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
          preActivations: JSON.parse(JSON.stringify(net.preActivations)),
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
    let timer: ReturnType<typeof setTimeout>;
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

      // Rotation Math
      const radH = (cameraAngle.h * Math.PI) / 180;
      const radV = (cameraAngle.v * Math.PI) / 180;
      
      const cosH = Math.cos(radH);
      const sinH = Math.sin(radH);
      const cosV = Math.cos(radV);
      const sinV = Math.sin(radV);
      
      // 1. Rotate around Y (Horizontal)
      const x1 = x3d * cosH - z3d * sinH;
      const y1 = y3d;
      const z1 = x3d * sinH + z3d * cosH;

      // 2. Rotate around X (Vertical)
      const x2 = x1;
      const y2 = y1 * cosV - z1 * sinV;
      const z2 = y1 * sinV + z1 * cosV;

      // Perspective vs Orthographic
      const fov = 1000;
      const baseScale = isOrtho ? 1.0 : (fov / (fov - z2)); 
      const finalScale = baseScale * viewScale;

      // Apply viewScale to coordinates for zoom translation effect
      return {
          x: 400 + (x2 * viewScale * (isOrtho ? 1 : baseScale)) + viewOffset.x, 
          y: 300 + (y2 * viewScale * (isOrtho ? 1 : baseScale)) + viewOffset.y, 
          scale: finalScale,
          depth: z2
      };
  };

  // --- Helpers ---
  const getActivationVisual = (key: ActivationKey, avgInput: number, width: number, height: number) => {
    if (!isFinite(avgInput)) avgInput = 0;
    const func = ACTIVATION_FUNCTIONS[key].func;
    const points = [];
    const range = 3; 
    const step = 0.3;
    const normX = (x: number) => (x + range) / (range * 2) * width;
    const normY = (y: number) => {
        let yNorm = y;
        if (key === 'SIGMOID') yNorm = y; 
        else if (key === 'TANH') yNorm = (y + 1) / 2; 
        else yNorm = (y + 1) / 3; 
        return height - (Math.max(0, Math.min(1, yNorm)) * height);
    };
    for (let x = -range; x <= range; x += step) {
        const y = func(x);
        points.push(`${normX(x).toFixed(1)},${normY(y).toFixed(1)}`);
    }
    const pathD = `M ${points.join(' L ')}`;
    const dotX = normX(Math.max(-range, Math.min(range, avgInput)));
    const dotY = normY(func(avgInput));
    return { pathD, dotX, dotY };
  };

  const cycleActivation = (layerIdx: number) => {
      const current = layerActivations[layerIdx];
      const currentIndex = ACTIVATION_KEYS.indexOf(current);
      const nextIndex = (currentIndex + 1) % ACTIVATION_KEYS.length;
      const nextKey = ACTIVATION_KEYS[nextIndex];
      
      const newActivations = [...layerActivations];
      newActivations[layerIdx] = nextKey;
      setLayerActivations(newActivations);
      setIsOptimized(false);
  };

  // --- Axis Gizmo Helpers ---
  const renderAxisGizmo = () => {
      const radH = (cameraAngle.h * Math.PI) / 180;
      const radV = (cameraAngle.v * Math.PI) / 180;
      const cosH = Math.cos(radH);
      const sinH = Math.sin(radH);
      const cosV = Math.cos(radV);
      const sinV = Math.sin(radV);

      const projectAxis = (x: number, y: number, z: number) => {
          const x1 = x * cosH - z * sinH;
          const y1 = y;
          const z1 = x * sinH + z * cosH;

          const x2 = x1;
          const y2 = y1 * cosV - z1 * sinV;
          const z2 = y1 * sinV + z1 * cosV;

          return { x: 50 + x2 * 35, y: 50 + y2 * 35, z: z2 }; // Center at 50,50 in gizmo svg
      };

      const origin = projectAxis(0, 0, 0);
      const xAxis = projectAxis(1, 0, 0); // Layer Axis (Red)
      const yAxis = projectAxis(0, 1, 0); // Row Axis (Green)
      const zAxis = projectAxis(0, 0, 1); // Col Axis (Blue)

      const handleAxisClick = (e: React.MouseEvent, view: {h: number, v: number}) => {
          e.stopPropagation();
          const isSameView = Math.abs(cameraAngle.h - view.h) < 0.1 && Math.abs(cameraAngle.v - view.v) < 0.1;
          
          if (isSameView) {
              setIsOrtho(!isOrtho);
          } else {
              setCameraAngle(view);
              setIsOrtho(false);
          }
      };

      const items = [
          { id: 'x', color: '#ff4444', end: xAxis, label: 'X', view: { h: 90, v: 0 } },
          { id: 'y', color: '#44ff44', end: yAxis, label: 'Y', view: { h: 0, v: 90 } },
          { id: 'z', color: '#4444ff', end: zAxis, label: 'Z', view: { h: 0, v: 0 } }
      ].sort((a, b) => a.end.z - b.end.z); // Sort by depth so front axes are on top

      return (
          <svg width="100" height="100" className="overflow-visible select-none">
              {items.map(item => (
                  <g key={item.id} className="cursor-pointer hover:brightness-125" onClick={(e) => handleAxisClick(e, item.view)}>
                      <line x1={origin.x} y1={origin.y} x2={item.end.x} y2={item.end.y} stroke={item.color} strokeWidth="2" />
                      <circle cx={item.end.x} cy={item.end.y} r="6" fill={item.color} />
                      <text x={item.end.x} y={item.end.y + 3} textAnchor="middle" fill="black" fontSize="8" fontWeight="bold">{item.label}</text>
                  </g>
              ))}
              <circle cx={origin.x} cy={origin.y} r="4" fill={isOrtho ? "#00f3ff" : "white"} className="cursor-pointer hover:fill-slate-300" onClick={(e) => { e.stopPropagation(); setCameraAngle({h: 25, v: 15}); setIsOrtho(false); }}/>
          </svg>
      );
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

                  // Activation Control (Attached to TOP of layer column) - Only for Hidden Layers
                  if (!isInput && !isOutput && r === 0 && c === Math.floor(dim.cols/2)) {
                       items.push({
                          type: 'ACTIVATION_CONTROL',
                          key: `act-${lIdx}`,
                          lIdx,
                          pos: { ...pos, y: pos.y - 90 * pos.scale }, // Position above
                          depth: pos.depth + 10 
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
  }, [layerDims, cameraAngle, viewScale, viewOffset, displayStats, direction, activeLayer, inputs, target, viewSettings, isOrtho]);

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
  const handleStartUIDrag = (e: React.MouseEvent, id: 'LOSS' | 'SETTINGS') => {
      e.stopPropagation();
      const rect = e.currentTarget.getBoundingClientRect();
      draggingUIRef.current = {
          id,
          offsetX: e.clientX - rect.left,
          offsetY: e.clientY - rect.top
      };
  };

  const handleMouseDownCanvas = (e: React.MouseEvent) => { 
      isDraggingCanvasRef.current = true; 
      lastMouseRef.current = { x: e.clientX, y: e.clientY }; 
  };

  const handleMouseMove = (e: React.MouseEvent) => {
      // Handle UI Dragging
      if (draggingUIRef.current) {
          const { id, offsetX, offsetY } = draggingUIRef.current;
          const newX = e.clientX - offsetX;
          const newY = e.clientY - offsetY;
          
          if (id === 'LOSS') {
              setLossGraphPos({ x: Math.max(0, newX), y: Math.max(0, newY) });
          } else {
              setViewSettingsPos({ x: Math.max(0, newX), y: Math.max(0, newY) });
          }
          return;
      }

      // Handle Canvas Dragging/Orbit
      if (!isDraggingCanvasRef.current) return;
      
      const dx = e.clientX - lastMouseRef.current.x;
      const dy = e.clientY - lastMouseRef.current.y;
      
      if (e.buttons === 1) {
          // Left Click: Orbit (Yaw and Pitch)
          // Disables Ortho mode when manually rotating
          setIsOrtho(false);
          setCameraAngle(prev => ({
              h: prev.h + dx * 0.5,
              v: Math.max(-90, Math.min(90, prev.v + dy * 0.5))
          }));
      } else if (e.buttons === 2 || e.buttons === 3) {
          // Right Click or Both: Pan
          setViewOffset(prev => ({
              x: prev.x + dx,
              y: prev.y + dy
          }));
      }
      
      lastMouseRef.current = { x: e.clientX, y: e.clientY };
  };
  const handleMouseUp = () => { 
      isDraggingCanvasRef.current = false; 
      draggingUIRef.current = null;
  };
  const handleWheel = (e: React.WheelEvent) => { setViewScale(s => Math.max(0.05, Math.min(5, s - e.deltaY * 0.001))); };

  return (
    <div className="h-full flex flex-col p-6 relative select-none" onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
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

      {/* Loss Graph (Draggable) */}
      <div 
        style={{ left: lossGraphPos.x, top: lossGraphPos.y }}
        onMouseDown={(e) => handleStartUIDrag(e, 'LOSS')}
        className="absolute z-20 w-64 h-32 bg-panel-bg/80 backdrop-blur border border-white/10 rounded-xl p-2 shadow-xl cursor-auto"
      >
         <div className="flex justify-between items-center px-2 mb-1 cursor-move" title="Drag to move">
            <div className="flex items-center gap-2">
                <GripHorizontal size={14} className="text-slate-500"/>
                <span className="text-[10px] font-bold text-slate-400">LOSS</span>
            </div>
            <span className={`text-[10px] font-mono font-bold ${displayStats.loss < 0.01 ? 'text-green-400' : 'text-red-400'}`}>{displayStats.loss.toFixed(6)}</span>
         </div>
         <div onMouseDown={e => e.stopPropagation()} className="h-[80%]">
             <ResponsiveContainer width="100%" height="100%">
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
      </div>

      {/* VIEW SETTINGS PANEL (Draggable, Default Left) */}
      <div 
          style={{ left: viewSettingsPos.x, top: viewSettingsPos.y }}
          onMouseDown={(e) => handleStartUIDrag(e, 'SETTINGS')}
          className="absolute z-30 bg-panel-bg/80 backdrop-blur border border-white/10 rounded-xl p-4 shadow-xl w-48 flex flex-col gap-4"
      >
          <div className="flex items-center gap-2 text-slate-400 text-xs font-bold border-b border-white/10 pb-2 cursor-move">
              <GripHorizontal size={14} /> <Eye size={14} /> VIEW SETTINGS
          </div>
          
          <div onMouseDown={e => e.stopPropagation()}>
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">NODE SIZE</div>
              <input 
                  type="range" min="5" max="30" step="1" 
                  value={viewSettings.nodeSize} 
                  onChange={(e) => setViewSettings(s => ({...s, nodeSize: Number(e.target.value)}))}
                  className="w-full accent-blue-500 h-1 bg-slate-700 rounded"
              />
          </div>

          <div onMouseDown={e => e.stopPropagation()}>
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">TEXT SIZE</div>
              <input 
                  type="range" min="0" max="20" step="1" 
                  value={viewSettings.textSize} 
                  onChange={(e) => setViewSettings(s => ({...s, textSize: Number(e.target.value)}))}
                  className="w-full accent-blue-500 h-1 bg-slate-700 rounded"
              />
          </div>

          <div onMouseDown={e => e.stopPropagation()}>
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">ANIM WIDTH</div>
              <input 
                  type="range" min="1" max="15" step="1" 
                  value={viewSettings.animWidth} 
                  onChange={(e) => setViewSettings(s => ({...s, animWidth: Number(e.target.value)}))}
                  className="w-full accent-green-500 h-1 bg-slate-700 rounded"
              />
          </div>
          
          <div onMouseDown={e => e.stopPropagation()}>
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">SLIDER SIZE</div>
              <input 
                  type="range" min="0" max="2.0" step="0.1" 
                  value={viewSettings.sliderSize} 
                  onChange={(e) => setViewSettings(s => ({...s, sliderSize: Number(e.target.value)}))}
                  className="w-full accent-pink-500 h-1 bg-slate-700 rounded"
              />
          </div>
          
           <div onMouseDown={e => e.stopPropagation()}>
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">PROJECTION</div>
              <button 
                onClick={() => setIsOrtho(!isOrtho)}
                className={`w-full py-1 text-[10px] font-bold rounded border ${isOrtho ? 'bg-neon-blue/20 border-neon-blue text-neon-blue' : 'bg-slate-800 border-white/10 text-slate-400'}`}
              >
                  {isOrtho ? '2D ORTHOGRAPHIC' : '3D PERSPECTIVE'}
              </button>
          </div>
      </div>
      
      {/* 3D AXIS GIZMO (Unity Style) */}
      <div className="absolute top-4 right-6 z-30 bg-black/40 rounded-full p-1 border border-white/10 backdrop-blur">
          {renderAxisGizmo()}
      </div>

      {/* Bottom Controls */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-30 bg-panel-bg/90 border border-white/10 rounded-2xl p-4 flex items-center gap-6 backdrop-blur shadow-2xl">
         <button onClick={() => setIsPlaying(!isPlaying)} className={`w-12 h-12 rounded-xl flex items-center justify-center ${isPlaying ? 'bg-red-500 text-white' : 'bg-blue-500 text-white'}`}>
            {isPlaying ? <Pause size={24} fill="currentColor"/> : <Play size={24} fill="currentColor"/>}
         </button>
         <button onClick={resetSimulation} className="w-12 h-12 bg-slate-800 text-slate-400 hover:text-white rounded-xl flex items-center justify-center"><RotateCcw size={20}/></button>
         
         <div className="w-px h-10 bg-white/10" />
         
         <div className="flex flex-col w-48 gap-1">
            <div className="flex justify-between text-[10px] font-bold text-slate-500 uppercase">
                <span>Speed</span>
                <span className={simSpeed < 1 ? 'text-neon-green' : ''}>{simSpeed < 1 ? 'TURBO' : `${simSpeed.toFixed(1)}ms`}</span>
            </div>
            <input type="range" min="0.1" max="200" step="0.1" value={200.1 - simSpeed} onChange={(e) => setSimSpeed(200.1 - Number(e.target.value))} className="w-full accent-blue-500 h-1.5 bg-slate-700 rounded appearance-none"/>
         </div>

         <div className="w-px h-10 bg-white/10" />

         <div className="flex flex-col w-48 gap-1">
             <div className="flex justify-between text-[10px] font-bold text-slate-500 uppercase">
                <span>Learning Rate</span>
                <span>{learningRate.toFixed(2)}</span>
            </div>
             <input 
                type="range" min="0.01" max="1.0" step="0.01"
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-full accent-purple-500 h-1.5 bg-slate-700 rounded appearance-none"
            />
         </div>
      </div>
      
      {/* Visualization Canvas */}
      <div className="flex-1 bg-slate-950 rounded-xl border border-white/10 relative shadow-inner overflow-hidden">
         {isOptimized && (
            <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                <div className="bg-panel-bg border border-neon-green/50 rounded-2xl p-8 shadow-[0_0_50px_rgba(0,255,0,0.2)] max-w-md w-full flex flex-col items-center gap-6 animate-in fade-in zoom-in duration-300">
                    <div className="w-16 h-16 rounded-full bg-neon-green/20 flex items-center justify-center border border-neon-green text-neon-green shadow-[0_0_20px_rgba(0,255,0,0.4)]">
                        <CheckCircle size={32} />
                    </div>
                    
                    <div className="text-center">
                        <h2 className="text-2xl font-bold text-white font-mono tracking-wider">OPTIMIZATION COMPLETE</h2>
                        <p className="text-slate-400 text-sm mt-2">Target loss threshold reached.</p>
                    </div>

                    <div className="grid grid-cols-2 gap-4 w-full">
                        <div className="bg-slate-900/50 p-4 rounded-xl border border-white/5 text-center">
                            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Epochs</div>
                            <div className="text-2xl font-mono font-bold text-white">{epochs}</div>
                        </div>
                        <div className="bg-slate-900/50 p-4 rounded-xl border border-white/5 text-center">
                            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Final Loss</div>
                            <div className="text-2xl font-mono font-bold text-neon-green">{displayStats.loss.toFixed(6)}</div>
                        </div>
                        <div className="bg-slate-900/50 p-4 rounded-xl border border-white/5 text-center">
                            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Learning Rate</div>
                            <div className="text-xl font-mono font-bold text-white">{learningRate}</div>
                        </div>
                        <div className="bg-slate-900/50 p-4 rounded-xl border border-white/5 text-center">
                            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Structure</div>
                            <div className="text-xs font-mono text-slate-300 mt-1">{layerDims.map(d => `${d.rows}x${d.cols}`).join(' - ')}</div>
                        </div>
                    </div>

                    <button 
                        onClick={() => setIsOptimized(false)}
                        className="w-full py-3 bg-neon-green text-black font-bold rounded-xl hover:bg-white transition-colors flex items-center justify-center gap-2"
                    >
                        CONTINUE TRAINING
                    </button>
                </div>
            </div>
         )}
         <svg 
            ref={svgRef}
            className="w-full h-full cursor-move"
            onMouseDown={handleMouseDownCanvas} 
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
                     const fontSize = Math.max(10, viewSettings.textSize) * pos.scale;

                     const fillOpacity = isInput ? 1 : 0.2 + (value * 0.8);
                     
                     const sSize = viewSettings.sliderSize;

                     // Layout calculations
                     const textY = pos.y + r + (fontSize * 0.5);
                     const biasTextY = textY + fontSize;
                     const biasSliderY = biasTextY + (fontSize * 0.5) + (5 * pos.scale);
                     
                     // Top Slider (Inputs/Targets)
                     const topFOHeight = 60 * pos.scale * sSize; // Increased height to fix clipping
                     const topFOWidth = 100 * pos.scale * sSize;
                     const topFOY = pos.y - r - topFOHeight - (5 * pos.scale); // Position above node with buffer
                     const topFOX = pos.x - (topFOWidth / 2);

                     return (
                         <g key={key}>
                             <circle cx={pos.x} cy={pos.y} r={r} 
                                fill={isActive ? (direction==='FORWARD'?'#00ff00':'#ff0000') : '#00f3ff'} 
                                fillOpacity={Math.min(1, Math.max(0, fillOpacity))} 
                                stroke={isActive ? (direction==='FORWARD'?'#00ff00':'#ff0000') : '#00f3ff'}
                                strokeWidth={2 * pos.scale}
                                filter={isActive ? `url(#glow-${direction==='FORWARD'?'green':'red'})` : ''}
                             />
                             
                             {/* Node Value (Activation) - Bottom of Node */}
                             {pos.scale > 0.4 && viewSettings.textSize > 0 && (
                                 <text 
                                    x={pos.x} 
                                    y={textY} 
                                    textAnchor="middle" 
                                    fill="white" 
                                    fontSize={fontSize} 
                                    className="font-mono font-bold pointer-events-none"
                                    style={{ textShadow: '0px 1px 4px rgba(0,0,0,0.8)' }}
                                 >
                                     {value.toFixed(2)}
                                 </text>
                             )}
                             
                             {/* Bias Value Text - Below Activation */}
                             {!isInput && !isOutput && pos.scale > 0.4 && viewSettings.textSize > 0 && (
                                 <text 
                                    x={pos.x} 
                                    y={biasTextY} 
                                    textAnchor="middle" 
                                    fill="#d8b4fe"
                                    fontSize={fontSize * 0.8} 
                                    className="font-mono pointer-events-none"
                                    style={{ textShadow: '0px 1px 4px rgba(0,0,0,0.8)' }}
                                 >
                                     b:{bias.toFixed(2)}
                                 </text>
                             )}

                             {/* Interactive Inputs/Targets - ABOVE Node */}
                             {(isInput || isOutput) && sSize > 0 && (
                                 <foreignObject 
                                     x={topFOX} 
                                     y={topFOY} 
                                     width={topFOWidth} 
                                     height={topFOHeight} 
                                     style={{
                                         // Ensure smooth transforms
                                     }}
                                 >
                                     <div className="flex flex-col items-center justify-center rounded h-full" onMouseDown={(e) => e.stopPropagation()}>
                                         <input 
                                            type="range" min="0" max="1" step="0.1" 
                                            value={isInput ? (inputs[flatIdx]||0) : (target[flatIdx]||0)}
                                            onChange={(e) => {
                                                const val = Number(e.target.value);
                                                if (isInput) { const n=[...inputs]; n[flatIdx]=val; setInputs(n); }
                                                else { const n=[...target]; n[flatIdx]=val; setTarget(n); }
                                            }}
                                            className={`rounded appearance-none ${isInput ? 'bg-blue-600 accent-blue-400' : 'bg-amber-600 accent-amber-400'}`}
                                            style={{
                                                width: '80%',
                                                height: '15%'
                                            }}
                                         />
                                     </div>
                                 </foreignObject>
                             )}

                             {/* Bias Slider - BELOW Text */}
                             {!isInput && !isOutput && pos.scale > 0.5 && sSize > 0 && (
                                  <foreignObject 
                                    x={pos.x - (30 * pos.scale * sSize)} 
                                    y={biasSliderY} 
                                    width={60 * pos.scale * sSize} 
                                    height={30 * pos.scale * sSize} 
                                  >
                                      <div className="flex justify-center items-center h-full" onMouseDown={(e) => e.stopPropagation()}>
                                          <input 
                                            type="range" min="-1" max="1" step="0.1" value={bias || 0}
                                            onChange={(e) => networkRef.current?.setLayerBias(lIdx, Number(e.target.value))}
                                            className="w-full h-1 bg-purple-900 accent-purple-500 rounded appearance-none opacity-80 hover:opacity-100 cursor-pointer"
                                            style={{ height: '20%' }}
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
                 } else if (item.type === 'ACTIVATION_CONTROL') {
                    const { key, pos, lIdx } = item;
                    const actKey = layerActivations[lIdx];
                    const actDef = ACTIVATION_FUNCTIONS[actKey];
                    
                    // Calculate average pre-activation for visual dot
                    const layerStartIdx = displayStats.values.slice(0, lIdx).reduce((acc, val) => acc + val.length, 0);
                    const layerPreActs = displayStats.preActivations[lIdx] || [];
                    const avgPreAct = layerPreActs.length > 0 
                        ? layerPreActs.reduce((a, b) => a + b, 0) / layerPreActs.length
                        : 0;

                    const visual = getActivationVisual(actKey, avgPreAct, 40, 20);

                    return (
                        <g key={key} transform={`translate(${pos.x}, ${pos.y}) scale(${pos.scale})`} className="cursor-pointer hover:opacity-80" onClick={() => cycleActivation(lIdx)} onMouseDown={(e) => e.stopPropagation()}>
                            <rect x="-30" y="-20" width="60" height="40" rx="8" fill="#1e293b" stroke={actDef.color} strokeWidth="1.5" fillOpacity="0.9" />
                            <text x="0" y="-8" textAnchor="middle" fill={actDef.color} fontSize="8" fontWeight="bold" fontFamily="monospace">{actDef.name}</text>
                            
                            {/* Mini Graph */}
                            <g transform="translate(-20, 0)">
                                <path d={visual.pathD} fill="none" stroke={actDef.color} strokeWidth="1.5" />
                                <circle cx={visual.dotX} cy={visual.dotY} r="2" fill="white" className="animate-pulse" />
                            </g>
                        </g>
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