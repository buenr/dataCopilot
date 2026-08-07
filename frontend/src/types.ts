export type ConnectionState = 'connecting' | 'connected' | 'offline' | 'error';
export type CanvasTab = 'web' | 'document' | 'empty';

export type DatasetColumn = {
  name: string;
  type: string;
  nullPercentage?: number;
};

export type Dataset = {
  id: string;
  name: string;
  kind: string;
  rows: number;
  columns: number;
  size?: string;
  variable?: string;
  schema: DatasetColumn[];
  sampleRows?: Array<Record<string, unknown>>;
};

export type ChatMessage = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  streaming?: boolean;
};

export type ToolRun = {
  id: string;
  name: string;
  status: 'running' | 'complete' | 'error';
  detail?: string;
  startedAt: number;
};

export type ExecutionState = {
  stdout: string;
  stderr: string;
  code: string;
  variables: Array<{ name: string; type: string; value?: string }>;
  running: boolean;
};

export type Artifact = {
  type?: 'webapp' | 'pdf' | 'document';
  name?: string;
  url?: string;
  port?: number;
  path?: string;
  title?: string;
};

export type SessionEvent = {
  type: string;
  [key: string]: unknown;
};
