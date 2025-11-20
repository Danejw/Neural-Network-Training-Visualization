export enum GameMode {
  MENU = 'MENU',
  FACTORY = 'FACTORY', // Neural Factory (Structure)
  PRISM = 'PRISM',     // Prism of Ancients (Vectors)
  DESCENT = 'DESCENT',  // Cyber Defense (Basic Optimization)
  QUANTUM = 'QUANTUM',   // Quantum Valley (Advanced Optimization)
  ARCHITECT = 'ARCHITECT' // Neural Architect (Full Network Builder)
}

export interface TutorContext {
  gameMode: GameMode;
  currentStats: Record<string, any>;
  userQuery?: string;
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}