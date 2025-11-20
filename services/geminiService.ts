import { GoogleGenAI } from "@google/genai";
import { TutorContext, GameMode } from '../types';

const apiKey = process.env.API_KEY || '';

// Initialize immediately - in a real app we'd handle missing keys more gracefully in UI
const ai = new GoogleGenAI({ apiKey });

export const getTutorExplanation = async (context: TutorContext): Promise<string> => {
  if (!apiKey) return "API Key is missing. Please configure process.env.API_KEY.";

  const { gameMode, currentStats } = context;

  let systemPrompt = `You are an expert AI Tutor inside a game called "Neural Nexus". 
  Your goal is to explain neural network concepts based on the current mini-game state.
  Keep explanations concise (under 80 words), encouraging, and use the game's metaphors.
  `;

  let gameContext = "";

  switch (gameMode) {
    case GameMode.FACTORY:
      systemPrompt += `
      Metaphor: Factory (Neural Net Structure). 
      Elements: Valves (Weights), Machines (Nodes), Sparking (Error).
      Concept: Forward propagation and adjusting weights to match output.
      `;
      gameContext = `Current Factory State: Input=${currentStats.input}, Weight=${currentStats.weight}, Bias=${currentStats.bias}, Output=${currentStats.output}, Target=${currentStats.target}.`;
      break;
    case GameMode.PRISM:
      systemPrompt += `
      Metaphor: Magic Prism (Linear Algebra). 
      Elements: Crystals (Matrices), Light Beam (Vector).
      Concept: Matrix multiplication rotating a vector. Dot product alignment.
      `;
      gameContext = `Current Prism State: Beam Angle=${currentStats.angle}deg, Target Angle=${currentStats.targetAngle}deg, Alignment Score=${currentStats.alignment}%.`;
      break;
    case GameMode.DESCENT:
      systemPrompt += `
      Metaphor: Cyber Defense (Gradient Descent).
      Elements: Landscape (Loss Function), Ball (Current State), Gravity (Gradient).
      Concept: Finding the global minimum (lowest error) by rolling down the slope.
      `;
      gameContext = `Current Descent State: Position X=${currentStats.x}, Loss=${currentStats.loss}, Learning Rate=${currentStats.learningRate}, Gradient=${currentStats.gradient}.`;
      break;
    case GameMode.QUANTUM:
      systemPrompt += `
      Metaphor: Quantum Valley (Advanced Optimization).
      Elements: Rugged Terrain (Non-convex Loss Function), Traps (Local Minima), Momentum (Velocity).
      Concept: Why simple descent gets stuck in small valleys (local minima) and how momentum helps escape to find the true bottom (global minimum).
      `;
      gameContext = `Current Valley State: Position=(${currentStats.x}, ${currentStats.z}), Loss=${currentStats.loss}, Momentum=${currentStats.momentum}, Stuck in Local Minima=${currentStats.isStuck}.`;
      break;
    case GameMode.ARCHITECT:
      systemPrompt += `
      Metaphor: The Neural Architect (Deep Learning).
      Elements: Layers (Depth), Nodes (Width), Synapses (Connections), Training Loop (Backprop).
      Concept: Designing a network architecture. How hidden layers extract features. The balance between network size and learning speed.
      `;
      gameContext = `Current Network State: Structure=[${currentStats.structure}], Total Epochs=${currentStats.epochs}, Current Loss=${currentStats.loss}, Learning Rate=${currentStats.lr}.`;
      break;
  }

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: `Context: ${gameContext}. User Question: Explain what is happening right now and how to win.`,
      config: {
        systemInstruction: systemPrompt,
      }
    });

    return response.text || "I'm analyzing the data stream...";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Connection to AI Tutor disrupted. Try again.";
  }
};