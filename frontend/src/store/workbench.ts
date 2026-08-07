import { create } from 'zustand';
import type {
  Artifact,
  CanvasTab,
  ChatMessage,
  ConnectionState,
  Dataset,
  ExecutionState,
  ToolRun,
} from '../types';

export type WorkbenchState = {
  sessionId?: string;
  connection: ConnectionState;
  messages: ChatMessage[];
  datasets: Dataset[];
  tools: ToolRun[];
  artifact?: Artifact;
  canvasTab: CanvasTab;
  canvasFullscreen: boolean;
  inspectorOpen: boolean;
  execution: ExecutionState;
  setSessionId: (sessionId: string) => void;
  setConnection: (connection: ConnectionState) => void;
  addMessage: (message: ChatMessage) => void;
  appendAssistantDelta: (delta: string) => void;
  finishAssistant: () => void;
  addDataset: (dataset: Dataset) => void;
  setArtifact: (artifact: Artifact) => void;
  setCanvasTab: (tab: CanvasTab) => void;
  setCanvasFullscreen: (value: boolean) => void;
  setInspectorOpen: (value: boolean) => void;
  startTool: (tool: ToolRun) => void;
  finishTool: (id: string, status?: ToolRun['status'], detail?: string) => void;
  updateExecution: (patch: Partial<ExecutionState>) => void;
};

const defaultExecution: ExecutionState = {
  stdout: '',
  stderr: '',
  code: '',
  variables: [],
  running: false,
};

export const useWorkbench = create<WorkbenchState>((set) => ({
  connection: 'connecting',
  messages: [],
  datasets: [],
  tools: [],
  canvasTab: 'empty',
  canvasFullscreen: false,
  inspectorOpen: false,
  execution: defaultExecution,
  setSessionId: (sessionId) => set({ sessionId }),
  setConnection: (connection) => set({ connection }),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  appendAssistantDelta: (delta) =>
    set((state) => {
      const last = state.messages[state.messages.length - 1];
      if (last?.role === 'assistant' && last.streaming) {
        return {
          messages: state.messages.map((message, index) =>
            index === state.messages.length - 1
              ? { ...message, content: message.content + delta }
              : message,
          ),
        };
      }
      return {
        messages: [
          ...state.messages,
          { id: `assistant-${Date.now()}`, role: 'assistant', content: delta, timestamp: Date.now(), streaming: true },
        ],
      };
    }),
  finishAssistant: () =>
    set((state) => ({
      messages: state.messages.map((message, index) =>
        index === state.messages.length - 1 ? { ...message, streaming: false } : message,
      ),
    })),
  addDataset: (dataset) => set((state) => ({ datasets: [...state.datasets, dataset] })),
  setArtifact: (artifact) =>
    set({
      artifact,
      canvasTab: artifact.type === 'pdf' || artifact.type === 'document' ? 'document' : 'web',
    }),
  setCanvasTab: (canvasTab) => set({ canvasTab }),
  setCanvasFullscreen: (canvasFullscreen) => set({ canvasFullscreen }),
  setInspectorOpen: (inspectorOpen) => set({ inspectorOpen }),
  startTool: (tool) => set((state) => ({ tools: [...state.tools, tool] })),
  finishTool: (id, status = 'complete', detail) =>
    set((state) => ({
      tools: state.tools.map((tool) => (tool.id === id ? { ...tool, status, detail } : tool)),
    })),
  updateExecution: (patch) => set((state) => ({ execution: { ...state.execution, ...patch } })),
}));
