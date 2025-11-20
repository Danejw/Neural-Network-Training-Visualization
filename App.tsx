import React, { useState, useCallback } from 'react';
import { FactoryGame } from './components/FactoryGame';
import { PrismGame } from './components/PrismGame';
import { DescentGame } from './components/DescentGame';
import { QuantumGame } from './components/QuantumGame';
import { NetworkGame } from './components/NetworkGame';
import { GeminiTutor } from './components/GeminiTutor';
import { GameMode, TutorContext } from './types';
import { BrainCircuit, Play, ChevronRight, Bot } from 'lucide-react';

const App: React.FC = () => {
  const [mode, setMode] = useState<GameMode>(GameMode.MENU);
  const [tutorOpen, setTutorOpen] = useState(false);
  const [tutorContext, setTutorContext] = useState<TutorContext>({
    gameMode: GameMode.MENU,
    currentStats: {}
  });

  const handleContextUpdate = useCallback((ctx: TutorContext) => {
    setTutorContext(ctx);
  }, []);

  const renderGame = () => {
    switch (mode) {
      case GameMode.FACTORY:
        return <FactoryGame onUpdateContext={handleContextUpdate} />;
      case GameMode.PRISM:
        return <PrismGame onUpdateContext={handleContextUpdate} />;
      case GameMode.DESCENT:
        return <DescentGame onUpdateContext={handleContextUpdate} />;
      case GameMode.QUANTUM:
        return <QuantumGame onUpdateContext={handleContextUpdate} />;
      case GameMode.ARCHITECT:
        return <NetworkGame onUpdateContext={handleContextUpdate} />;
      default:
        return (
          <div className="h-full flex items-center justify-center p-8 overflow-y-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl w-full">
              <GameCard 
                title="The Neural Factory" 
                subtitle="Structure & Weights"
                desc="Manage valves and pistons to understand how basic neurons process inputs."
                color="border-orange-500/50 hover:border-orange-500"
                onClick={() => setMode(GameMode.FACTORY)} 
              />
              <GameCard 
                title="Prism of Ancients" 
                subtitle="Vectors & Matrices"
                desc="Rotate magical crystals to grasp matrix multiplication and vector alignment."
                color="border-neon-pink/50 hover:border-neon-pink"
                onClick={() => setMode(GameMode.PRISM)} 
              />
              <GameCard 
                title="Cyber-Defense" 
                subtitle="Basic Gradient Descent"
                desc="Roll down a simple 3D Loss Landscape to optimize defense systems and minimize error."
                color="border-neon-green/50 hover:border-neon-green"
                onClick={() => setMode(GameMode.DESCENT)} 
              />
              <GameCard 
                title="Quantum Valley" 
                subtitle="Advanced Optimization"
                desc="Navigate a complex, randomized 3D terrain. Use Momentum to escape local minima traps."
                color="border-indigo-500/50 hover:border-indigo-500"
                onClick={() => setMode(GameMode.QUANTUM)} 
              />
               <GameCard 
                title="Neural Architect" 
                subtitle="Deep Learning Builder"
                desc="Build, train, and visualize a multi-layer neural network in real-time."
                color="border-blue-500/50 hover:border-blue-500"
                onClick={() => setMode(GameMode.ARCHITECT)} 
              />
            </div>
          </div>
        );
    }
  };

  return (
    <div className="h-screen flex flex-col bg-dark-bg text-white overflow-hidden font-sans selection:bg-neon-blue selection:text-black">
      {/* Top Navigation */}
      <nav className="h-16 border-b border-white/10 flex items-center justify-between px-6 bg-panel-bg z-20">
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => setMode(GameMode.MENU)}>
          <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
            <BrainCircuit size={24} className="text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg tracking-tight">NEURAL NEXUS</h1>
            <div className="text-[10px] text-slate-400 font-mono uppercase tracking-widest">Gamified Deep Learning Labs</div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {mode !== GameMode.MENU && (
            <button 
              onClick={() => setTutorOpen(!tutorOpen)}
              className={`flex items-center gap-2 px-4 py-2 rounded-full border transition-all ${tutorOpen ? 'bg-neon-blue/20 border-neon-blue text-neon-blue shadow-[0_0_15px_rgba(0,243,255,0.3)]' : 'border-white/20 text-slate-300 hover:bg-white/5'}`}
            >
              <Bot size={18} />
              <span className="text-sm font-bold">AI TUTOR</span>
            </button>
          )}
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 relative overflow-hidden">
        {renderGame()}
        <GeminiTutor context={tutorContext} isOpen={tutorOpen} onClose={() => setTutorOpen(false)} />
      </main>
    </div>
  );
};

// Helper Component for Menu
const GameCard = ({ title, subtitle, desc, color, onClick }: any) => (
  <button 
    onClick={onClick}
    className={`group relative h-64 bg-panel-bg rounded-2xl border ${color} p-8 text-left transition-all hover:-translate-y-2 hover:shadow-2xl overflow-hidden flex flex-col`}
  >
    <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
    <div className="mb-auto">
      <div className="text-xs font-mono text-slate-400 mb-2 uppercase tracking-widest">{subtitle}</div>
      <h3 className="text-2xl font-bold mb-3 text-white group-hover:text-neon-blue transition-colors">{title}</h3>
      <p className="text-slate-400 leading-relaxed text-sm">{desc}</p>
    </div>
    <div className="flex items-center gap-2 text-white font-bold text-xs uppercase tracking-wider group-hover:gap-4 transition-all">
      Initialize Sim <ChevronRight size={16} />
    </div>
  </button>
);

export default App;