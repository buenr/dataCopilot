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
  artifacts: Artifact[];
  canvasTab: CanvasTab;
  canvasFullscreen: boolean;
  inspectorOpen: boolean;
  execution: ExecutionState;
  thinking: boolean;
  turnActive: boolean;
  setSessionId: (sessionId: string) => void;
  setConnection: (connection: ConnectionState) => void;
  addMessage: (message: ChatMessage) => void;
  setMessages: (messages: ChatMessage[]) => void;
  appendAssistantDelta: (delta: string) => void;
  setAssistantMessage: (content: string) => void;
  finishAssistant: () => void;
  addDataset: (dataset: Dataset) => void;
  setDatasets: (datasets: Dataset[]) => void;
  removeDataset: (id: string) => void;
  setArtifact: (artifact: Artifact) => void;
  selectArtifact: (name: string) => void;
  setCanvasTab: (tab: CanvasTab) => void;
  setCanvasFullscreen: (value: boolean) => void;
  setInspectorOpen: (value: boolean) => void;
  startTool: (tool: ToolRun) => void;
  finishTool: (id: string, status?: ToolRun['status'], detail?: string) => void;
  clearTools: () => void;
  updateExecution: (patch: Partial<ExecutionState>) => void;
  appendExecutionOutput: (stream: 'stdout' | 'stderr', data: string) => void;
  setThinking: (value: boolean) => void;
  setTurnActive: (value: boolean) => void;
  reset: () => void;
};

const artifactKey = (artifact: Artifact) => artifact.name ?? artifact.path ?? artifact.url ?? '';

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
  artifacts: [],
  canvasTab: 'empty',
  canvasFullscreen: false,
  inspectorOpen: false,
  execution: defaultExecution,
  thinking: false,
  turnActive: false,
  setSessionId: (sessionId) => set({ sessionId }),
  setConnection: (connection) => set({ connection }),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  setMessages: (messages) => set({ messages }),
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
  setAssistantMessage: (content) =>
    set((state) => {
      const last = state.messages[state.messages.length - 1];
      if (last?.role === 'assistant' && last.streaming) {
        // The backend streams deltas and then re-sends the full reply in the
        // final assistant_message event; replace rather than append so the
        // reply is not duplicated.
        return {
          messages: state.messages.map((message, index) =>
            index === state.messages.length - 1 ? { ...message, content } : message,
          ),
        };
      }
      return {
        messages: [
          ...state.messages,
          { id: `assistant-${Date.now()}`, role: 'assistant', content, timestamp: Date.now(), streaming: true },
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
  setDatasets: (datasets) => set({ datasets }),
  removeDataset: (id) => set((state) => ({ datasets: state.datasets.filter((dataset) => dataset.id !== id) })),
  setArtifact: (artifact) =>
    set((state) => {
      // Keep every artifact so earlier results can be reopened from the canvas;
      // a re-registered name replaces its previous entry.
      const key = artifactKey(artifact);
      const artifacts =
        key && state.artifacts.some((item) => artifactKey(item) === key)
          ? state.artifacts.map((item) => (artifactKey(item) === key ? artifact : item))
          : [...state.artifacts, artifact];
      return {
        artifact,
        artifacts,
        canvasTab: artifact.type === 'pdf' || artifact.type === 'document' ? 'document' : 'web',
      };
    }),
  selectArtifact: (name) =>
    set((state) => {
      const artifact = state.artifacts.find((item) => item.name === name);
      if (!artifact) return {};
      return {
        artifact,
        canvasTab: artifact.type === 'pdf' || artifact.type === 'document' ? 'document' : 'web',
      };
    }),
  setCanvasTab: (canvasTab) => set({ canvasTab }),
  setCanvasFullscreen: (canvasFullscreen) => set({ canvasFullscreen }),
  setInspectorOpen: (inspectorOpen) => set({ inspectorOpen }),
  startTool: (tool) => set((state) => ({ tools: [...state.tools, tool] })),
  finishTool: (id, status = 'complete', detail) =>
    set((state) => ({
      tools: state.tools.map((tool) => (tool.id === id ? { ...tool, status, detail } : tool)),
    })),
  clearTools: () => set({ tools: [] }),
  updateExecution: (patch) => set((state) => ({ execution: { ...state.execution, ...patch } })),
  appendExecutionOutput: (stream, data) =>
    set((state) => ({
      execution: { ...state.execution, [stream]: state.execution[stream] + data },
    })),
  setThinking: (thinking) => set({ thinking }),
  setTurnActive: (turnActive) => set({ turnActive }),
  reset: () =>
    set({
      messages: [],
      datasets: [],
      tools: [],
      artifacts: [],
      artifact: undefined,
      canvasTab: 'empty',
      execution: { ...defaultExecution },
      thinking: false,
      turnActive: false,
    }),
}));
