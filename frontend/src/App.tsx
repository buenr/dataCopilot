import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  AlertCircle,
  ArrowUp,
  BarChart3,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Code2,
  Database,
  Download,
  ExternalLink,
  FileBarChart,
  FileJson,
  FileSpreadsheet,
  FileText,
  Gauge,
  Globe2,
  HardDriveUpload,
  LayoutPanelLeft,
  Maximize2,
  Minimize2,
  PanelRight,
  Paperclip,
  Play,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Square,
  Table2,
  Terminal,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import {
  apiUrl,
  createSession,
  deleteDataset,
  deleteSession,
  getSession,
  uploadDataset,
  websocketUrl,
} from './lib/api';
import { useWorkbench } from './store/workbench';
import type { WorkbenchState } from './store/workbench';
import type { ChatMessage, Dataset, DatasetColumn, SessionEvent } from './types';

const suggestions = [
  { label: 'Explore my data', icon: Search },
  { label: 'Build a dashboard', icon: BarChart3 },
  { label: 'Find key insights', icon: Sparkles },
  { label: 'Create a report', icon: FileBarChart },
];

const newId = (prefix: string) => `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

function textValue(value: unknown, fallback = '') {
  return typeof value === 'string' ? value : fallback;
}

function fileIcon(kind: string) {
  const lower = kind.toLowerCase();
  if (lower.includes('json')) return FileJson;
  if (lower.includes('xls')) return FileSpreadsheet;
  if (lower.includes('parquet')) return Database;
  return Table2;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat('en-US', { notation: value > 999999 ? 'compact' : 'standard' }).format(value);
}

function profileToDataset(summary: Record<string, unknown>, file?: File): Dataset {
  const dtypes =
    summary.dtypes && typeof summary.dtypes === 'object'
      ? (summary.dtypes as Record<string, unknown>)
      : {};
  const nullPercentages =
    summary.null_percentages && typeof summary.null_percentages === 'object'
      ? (summary.null_percentages as Record<string, unknown>)
      : {};
  const columns: DatasetColumn[] = Object.entries(dtypes).map(([name, type]) => ({
    name,
    type: textValue(type, 'unknown'),
    nullPercentage: Number(nullPercentages[name] ?? 0),
  }));
  const name = textValue(summary.file, file?.name ?? 'dataset');
  return {
    id: name || newId('dataset'),
    name,
    kind: name.split('.').pop()?.toUpperCase() || 'DATA',
    rows: Number(summary.rows ?? 0),
    columns: Number(summary.columns ?? columns.length),
    variable: textValue(summary.name),
    size: file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : undefined,
    schema: columns,
    sampleRows: Array.isArray(summary.sample_rows)
      ? (summary.sample_rows as Array<Record<string, unknown>>)
      : undefined,
  };
}

type ExecutionStateVariable = { name: string; type: string; value?: string };

export default function App() {
  const {
    sessionId,
    connection,
    messages,
    datasets,
    tools,
    artifact,
    artifacts,
    canvasTab,
    canvasFullscreen,
    inspectorOpen,
    execution,
    thinking,
    turnActive,
    setSessionId,
    setConnection,
    addMessage,
    setMessages,
    appendAssistantDelta,
    setAssistantMessage,
    finishAssistant,
    setDatasets,
    setArtifact,
    selectArtifact,
    setCanvasTab,
    setCanvasFullscreen,
    setInspectorOpen,
    startTool,
    finishTool,
    clearTools,
    updateExecution,
    appendExecutionOutput,
    setThinking,
    setTurnActive,
    reset,
  } = useWorkbench();
  const socketRef = useRef<WebSocket | null>(null);
  const intentionalCloseRef = useRef(false);
  const [draft, setDraft] = useState('');
  const [sessionError, setSessionError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const [sessionKey, setSessionKey] = useState(0);

  const handleEvent = useCallback((event: SessionEvent) => {
    const wb = useWorkbench.getState();
    const payload = (event.data && typeof event.data === 'object' ? event.data : event) as SessionEvent;
    switch (event.type) {
      case 'session_ready': {
        wb.setConnection('connected');
        // A reconnect replays the session transcript and dataset profiles so
        // the browser does not lose context across a dropped socket or refresh.
        const replayMessages = (event as Record<string, unknown>).messages;
        if (Array.isArray(replayMessages)) {
          wb.setMessages(
            (replayMessages as Array<Record<string, unknown>>).map((entry, index) => ({
              id: `replay-${index}`,
              role: (entry.role === 'assistant' ? 'assistant' : 'user') as ChatMessage['role'],
              content: textValue(entry.content),
              timestamp: typeof entry.timestamp === 'number' ? entry.timestamp : Date.now(),
            })),
          );
        }
        const replayDatasets = (event as Record<string, unknown>).datasets;
        if (Array.isArray(replayDatasets)) {
          wb.setDatasets(
            (replayDatasets as Array<Record<string, unknown>>).map((summary) =>
              profileToDataset(summary),
            ),
          );
        }
        wb.clearTools();
        wb.updateExecution({ stdout: '', stderr: '', code: '', variables: [], running: false });
        wb.setThinking(false);
        wb.setTurnActive(false);
        break;
      }
      case 'assistant_delta':
        wb.setThinking(false);
        wb.appendAssistantDelta(textValue(payload.delta ?? payload.content ?? payload.text));
        break;
      case 'assistant_message':
        wb.setAssistantMessage(textValue(payload.content ?? payload.message ?? payload.text));
        wb.finishAssistant();
        break;
      case 'tool_start': {
        const id = textValue(payload.id, newId('tool'));
        wb.startTool({
          id,
          name: textValue(payload.name ?? payload.tool, 'Working'),
          status: 'running',
          detail: textValue(payload.input),
          startedAt: Date.now(),
        });
        wb.setThinking(false);
        wb.updateExecution({ running: true });
        break;
      }
      case 'tool_result': {
        const id = textValue(payload.id);
        if (id)
          wb.finishTool(
            id,
            payload.error ? 'error' : 'complete',
            textValue(payload.stdout ?? payload.output ?? payload.detail ?? payload.error),
          );
        break;
      }
      case 'execution': {
        const status = textValue(payload.status);
        if (status === 'running') {
          const stream = textValue(payload.stream);
          const data = textValue(payload.data);
          if (stream && data) {
            // Streaming delta — accumulate into the right buffer so multi-cell
            // output is not lost (previously each event overwrote stdout).
            wb.appendExecutionOutput(stream === 'stderr' ? 'stderr' : 'stdout', data);
          } else {
            // Initial event carrying the code before the first cell runs.
            wb.updateExecution({
              code: textValue(payload.code ?? payload.generated_code, ''),
              stdout: '',
              stderr: '',
              running: true,
            });
          }
        } else {
          // status === 'complete' — stdout/stderr may be absent if already
          // streamed; only overwrite what the backend actually sent.
          const patch: Partial<WorkbenchState['execution']> = { running: false };
          if (typeof payload.stdout === 'string') patch.stdout = payload.stdout;
          if (typeof payload.stderr === 'string') patch.stderr = payload.stderr;
          if (Array.isArray(payload.variables))
            patch.variables = payload.variables as ExecutionStateVariable[];
          if (typeof payload.code === 'string') patch.code = payload.code;
          wb.updateExecution(patch);
        }
        break;
      }
      case 'artifact': {
        const incoming = (payload.artifact ?? payload) as Record<string, unknown>;
        wb.setArtifact({
          type: (incoming.type ?? incoming.kind ?? 'webapp') as 'webapp' | 'pdf' | 'document',
          name: textValue(incoming.name),
          title: textValue(incoming.title),
          url: textValue(incoming.url),
          path: textValue(incoming.path),
          port: Number(incoming.port ?? 0) || undefined,
        });
        break;
      }
      case 'cancelled':
        wb.addMessage({
          id: newId('system'),
          role: 'system',
          content: textValue(payload.message, 'Run stopped.'),
          timestamp: Date.now(),
        });
        wb.updateExecution({ running: false });
        wb.setThinking(false);
        wb.setTurnActive(false);
        break;
      case 'error':
        wb.setConnection('error');
        wb.addMessage({
          id: newId('error'),
          role: 'system',
          content: textValue(payload.message ?? payload.error, 'The gateway returned an error.'),
          timestamp: Date.now(),
        });
        wb.updateExecution({ running: false });
        wb.setThinking(false);
        wb.setTurnActive(false);
        break;
      case 'done':
        wb.finishAssistant();
        wb.updateExecution({ running: false });
        wb.setThinking(false);
        wb.setTurnActive(false);
        break;
      default:
        break;
    }
  }, []);

  useEffect(() => {
    let disposed = false;
    intentionalCloseRef.current = false;
    let reconnectDelay = 1000;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    async function connect(existingId?: string) {
      let id = existingId;
      if (!id) {
        try {
          id = await createSession();
        } catch (error) {
          if (disposed || intentionalCloseRef.current) return;
          setConnection('offline');
          setSessionError(error instanceof Error ? error.message : 'Gateway unavailable');
          reconnectTimer = setTimeout(() => connect(), reconnectDelay);
          reconnectDelay = Math.min(reconnectDelay * 2, 30000);
          return;
        }
      }
      if (disposed) return;
      setSessionId(id);
      sessionStorage.setItem('datacopilot-session-id', id);

      const socket = new WebSocket(websocketUrl(`/ws/sessions/${id}`));
      socketRef.current = socket;

      socket.onopen = () => {
        if (!disposed && socketRef.current === socket) {
          reconnectDelay = 1000;
        }
      };
      socket.onmessage = (message) => {
        if (!disposed && socketRef.current === socket) {
          try {
            handleEvent(JSON.parse(message.data) as SessionEvent);
          } catch {
            addMessage({
              id: newId('error'),
              role: 'system',
              content: 'Received an unreadable event from the gateway.',
              timestamp: Date.now(),
            });
          }
        }
      };
      socket.onerror = () => {
        if (!disposed && socketRef.current === socket) setConnection('error');
      };
      socket.onclose = () => {
        if (disposed || socketRef.current !== socket) return;
        setConnection('offline');
        if (intentionalCloseRef.current) return;
        // Reconnect to the same session with exponential backoff.
        reconnectTimer = setTimeout(() => connect(id), reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 30000);
      };
    }

    const existingId = sessionStorage.getItem('datacopilot-session-id');
    if (existingId) {
      // Verify the session still exists before reconnecting; a reaped
      // container means we must start fresh.
      getSession(existingId)
        .then((response) => {
          if (disposed) return;
          if (response.ok) {
            void connect(existingId);
          } else {
            sessionStorage.removeItem('datacopilot-session-id');
            void connect();
          }
        })
        .catch(() => {
          if (!disposed) void connect();
        });
    } else {
      void connect();
    }

    return () => {
      disposed = true;
      clearTimeout(reconnectTimer);
      socketRef.current?.close();
    };
    // Re-run only when a new session is explicitly requested.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionKey, handleEvent]);

  const sendMessage = (content = draft.trim()) => {
    if (!content) return;
    addMessage({ id: newId('user'), role: 'user', content, timestamp: Date.now() });
    setDraft('');
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      setThinking(true);
      setTurnActive(true);
      socketRef.current.send(JSON.stringify({ type: 'user_message', content }));
    } else {
      addMessage({
        id: newId('system'),
        role: 'system',
        content: 'Not connected — your message was not sent. Reconnecting…',
        timestamp: Date.now(),
      });
    }
  };

  const stopRun = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'cancel' }));
    }
  };

  const handleUpload = async (files: File[]) => {
    if (!sessionId || files.length === 0) {
      if (!sessionId) setSessionError('Waiting for a session before uploading.');
      return;
    }
    setUploading(true);
    try {
      const response = await uploadDataset(sessionId, files);
      const summaries = Array.isArray(response.schemas)
        ? (response.schemas as Array<Record<string, unknown>>)
        : [];
      setDatasets(
        summaries.map((summary) => {
          const matchedFile = files.find((f) => f.name === textValue(summary.file));
          return profileToDataset(summary, matchedFile);
        }),
      );
    } catch (error) {
      setSessionError(error instanceof Error ? error.message : 'Could not upload this file.');
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDataset = async (name: string) => {
    if (!sessionId) return;
    try {
      const response = await deleteDataset(sessionId, name);
      const summaries = Array.isArray(response.schemas)
        ? (response.schemas as Array<Record<string, unknown>>)
        : [];
      setDatasets(summaries.map((summary) => profileToDataset(summary)));
    } catch (error) {
      setSessionError(error instanceof Error ? error.message : 'Could not remove this dataset.');
    }
  };

  const handleNewSession = async () => {
    intentionalCloseRef.current = true;
    socketRef.current?.close();
    const oldId = useWorkbench.getState().sessionId;
    if (oldId) await deleteSession(oldId);
    sessionStorage.removeItem('datacopilot-session-id');
    reset();
    setSessionError('');
    setSessionKey((k) => k + 1);
  };

  return (
    <div className={`app-shell ${canvasFullscreen ? 'canvas-is-fullscreen' : ''}`}>
      <Header connection={connection} sessionId={sessionId} onNewSession={handleNewSession} />
      {sessionError && (
        <div className="gateway-banner">
          <AlertCircle size={15} />
          <span>{sessionError}</span>
          <button aria-label="Dismiss notification" onClick={() => setSessionError('')}>
            <X size={15} />
          </button>
        </div>
      )}
      <main className="workspace">
        {!canvasFullscreen && sidebarOpen && (
          <Explorer
            datasets={datasets}
            uploading={uploading}
            dragOver={dragOver}
            setDragOver={setDragOver}
            onUpload={handleUpload}
            onDeleteDataset={handleDeleteDataset}
          />
        )}
        {!canvasFullscreen && (
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label={sidebarOpen ? 'Hide data explorer' : 'Show data explorer'}
          >
            {sidebarOpen ? <ChevronLeftIcon /> : <LayoutPanelLeft size={16} />}
          </button>
        )}
        <PanelGroup direction="horizontal" className="main-panels">
          {!canvasFullscreen && (
            <>
              <Panel defaultSize={35} minSize={28} maxSize={52} className="chat-panel">
                <Chat
                  messages={messages}
                  tools={tools}
                  draft={draft}
                  onDraft={setDraft}
                  onSend={() => sendMessage()}
                  onSuggestion={sendMessage}
                  onUpload={handleUpload}
                  inspectorOpen={inspectorOpen}
                  onToggleInspector={() => setInspectorOpen(!inspectorOpen)}
                  execution={execution}
                  connected={connection === 'connected'}
                  thinking={thinking}
                  turnActive={turnActive}
                  onStop={stopRun}
                />
              </Panel>
              <PanelResizeHandle className="resize-handle">
                <span />
              </PanelResizeHandle>
            </>
          )}
          <Panel defaultSize={65} minSize={35} className="canvas-panel">
            <Canvas
              sessionId={sessionId}
              artifact={artifact}
              artifacts={artifacts}
              tab={canvasTab}
              onTab={setCanvasTab}
              fullscreen={canvasFullscreen}
              onFullscreen={() => setCanvasFullscreen(!canvasFullscreen)}
              onSelectArtifact={selectArtifact}
            />
          </Panel>
        </PanelGroup>
      </main>
    </div>
  );
}

function ChevronLeftIcon() {
  return <ChevronRight size={16} className="rotate-180" />;
}

function Header({
  connection,
  sessionId,
  onNewSession,
}: {
  connection: string;
  sessionId?: string;
  onNewSession: () => void;
}) {
  const [helpOpen, setHelpOpen] = useState(false);
  const shortId = sessionId ? sessionId.slice(0, 8) : '--------';
  return (
    <header className="topbar">
      <div className="brand-lockup">
        <div className="brand-mark">
          <Sparkles size={16} />
        </div>
        <div>
          <div className="brand-name">
            data<span>copilot</span>
          </div>
          <div className="brand-caption">ANALYSIS WORKBENCH</div>
        </div>
      </div>
      <div className="session-meta">
        <span className="session-pill">
          <span className={`status-dot ${connection}`} />
          {connection === 'connected' ? 'Session active' : connection === 'connecting' ? 'Starting session' : 'Offline mode'}
        </span>
        <span className="session-id">
          SESSION / <b>{shortId}</b>
        </span>
      </div>
      <div className="top-actions">
        <button className="icon-button" aria-label="Help" aria-expanded={helpOpen} onClick={() => setHelpOpen((open) => !open)}>
          <CircleHelp size={17} />
        </button>
        <button className="new-session" onClick={onNewSession}>
          <Plus size={16} /> New session
        </button>
      </div>
      {helpOpen && (
        <div className="help-popover">
          <div className="help-popover-header">
            <strong>How Data Copilot works</strong>
            <button className="icon-button" aria-label="Close help" onClick={() => setHelpOpen(false)}>
              <X size={14} />
            </button>
          </div>
          <ul>
            <li>Upload CSV, XLSX, Parquet, or JSON from the data explorer or the attach button in the composer.</li>
            <li>Ask questions in chat; the agent profiles your data and runs Python in an isolated sandbox.</li>
            <li>Dashboards and reports render in the canvas. Generated code, output, and variables appear in the execution inspector.</li>
            <li>Sessions persist across page refreshes and expire after 30 minutes of inactivity.</li>
          </ul>
        </div>
      )}
    </header>
  );
}

function Explorer({
  datasets,
  uploading,
  dragOver,
  setDragOver,
  onUpload,
  onDeleteDataset,
}: {
  datasets: Dataset[];
  uploading: boolean;
  dragOver: boolean;
  setDragOver: (value: boolean) => void;
  onUpload: (files: File[]) => void;
  onDeleteDataset: (name: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <aside className="explorer">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Workspace</span>
          <h2>Data explorer</h2>
        </div>
      </div>
      <label
        className={`upload-zone ${uploading ? 'is-uploading' : ''} ${dragOver ? 'drag-over' : ''}`}
        htmlFor="dataset-upload"
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const files = Array.from(e.dataTransfer.files).filter((f) =>
            /\.(csv|xlsx|xls|parquet|json)$/i.test(f.name),
          );
          if (files.length) onUpload(files);
        }}
      >
        <div className="upload-icon">
          {uploading ? <RefreshCw className="spin" size={19} /> : <HardDriveUpload size={19} />}
        </div>
        <strong>{uploading ? 'Profiling dataset…' : 'Upload a dataset'}</strong>
        <span>CSV, XLSX, Parquet or JSON — drag or browse</span>
        <input
          ref={inputRef}
          id="dataset-upload"
          type="file"
          multiple
          accept=".csv,.xlsx,.xls,.parquet,.json"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            if (files.length) onUpload(files);
            event.target.value = '';
          }}
        />
      </label>
      <div className="explorer-label">
        <span>Datasets</span>
        <span className="count-badge">{datasets.length}</span>
      </div>
      <div className="dataset-list">
        {datasets.length === 0 ? (
          <div className="empty-explorer">
            <Database size={21} />
            <p>No datasets yet</p>
            <span>Upload data to start exploring.</span>
          </div>
        ) : (
          datasets.map((dataset) => (
            <DatasetCard key={dataset.id} dataset={dataset} onDelete={onDeleteDataset} />
          ))
        )}
      </div>
      <div className="explorer-footer">
        <div className="tiny-status">
          <span className="status-dot connected" /> Sandbox ready
        </div>
        <span className="mono">v0.1 POC</span>
      </div>
    </aside>
  );
}

function DatasetCard({ dataset, onDelete }: { dataset: Dataset; onDelete: (name: string) => void }) {
  const [open, setOpen] = useState(true);
  const [showAllColumns, setShowAllColumns] = useState(false);
  const [showSample, setShowSample] = useState(false);
  const Icon = fileIcon(dataset.kind);
  const visibleColumns = showAllColumns ? dataset.schema : dataset.schema.slice(0, 6);
  return (
    <div className={`dataset-card ${open ? 'open' : ''}`}>
      <button className="dataset-header" onClick={() => setOpen(!open)} aria-expanded={open}>
        <div className="file-icon">
          <Icon size={17} />
        </div>
        <div className="dataset-title">
          <strong>{dataset.name}</strong>
          <span>
            {dataset.variable || 'dataset'} <i>·</i> {dataset.kind}
          </span>
        </div>
        {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
      </button>
      {open && (
        <div className="dataset-body">
          <div className="dataset-stats">
            <span>
              <b>{formatNumber(dataset.rows)}</b> rows
            </span>
            <span>
              <b>{dataset.columns}</b> cols
            </span>
            {dataset.size && (
              <span>
                <b>{dataset.size}</b>
              </span>
            )}
            <button
              className="dataset-delete"
              aria-label="Remove dataset"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(dataset.name);
              }}
            >
              <X size={13} />
            </button>
          </div>
          {dataset.schema.length > 0 && (
            <div className="schema-list">
              {visibleColumns.map((column) => (
                <div className="schema-row" key={column.name}>
                  <span className="schema-type">
                    {column.type === 'object'
                      ? 'Aa'
                      : column.type.includes('int') || column.type.includes('float')
                        ? '#'
                        : '◷'}
                  </span>
                  <span>{column.name}</span>
                  {column.nullPercentage ? <em>{column.nullPercentage}% null</em> : null}
                </div>
              ))}
            </div>
          )}
          {dataset.schema.length > 6 && (
            <button className="more-columns" onClick={() => setShowAllColumns(!showAllColumns)}>
              {showAllColumns ? 'Show fewer' : `+ ${dataset.schema.length - 6} more columns`}
            </button>
          )}
          {dataset.sampleRows && dataset.sampleRows.length > 0 && (
            <>
              <button className="more-columns" onClick={() => setShowSample(!showSample)}>
                {showSample ? 'Hide sample rows' : `Show ${dataset.sampleRows.length} sample rows`}
              </button>
              {showSample && (
                <div className="sample-rows">
                  <table>
                    <thead>
                      <tr>
                        {dataset.schema.slice(0, 8).map((col) => (
                          <th key={col.name}>{col.name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {dataset.sampleRows.map((row, i) => (
                        <tr key={i}>
                          {dataset.schema.slice(0, 8).map((col) => (
                            <td key={col.name}>{String(row[col.name] ?? '')}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function Chat({
  messages,
  tools,
  draft,
  onDraft,
  onSend,
  onSuggestion,
  onUpload,
  inspectorOpen,
  onToggleInspector,
  execution,
  connected,
  thinking,
  turnActive,
  onStop,
}: {
  messages: WorkbenchState['messages'];
  tools: WorkbenchState['tools'];
  draft: string;
  onDraft: (value: string) => void;
  onSend: () => void;
  onSuggestion: (value: string) => void;
  onUpload: (files: File[]) => void;
  inspectorOpen: boolean;
  onToggleInspector: () => void;
  execution: WorkbenchState['execution'];
  connected: boolean;
  thinking: boolean;
  turnActive: boolean;
  onStop: () => void;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  const attachRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, tools]);
  return (
    <section className="chat-workbench">
      <div className="column-header">
        <div>
          <span className="eyebrow">Copilot</span>
          <h1>What will we uncover?</h1>
        </div>
      </div>
      <div className="chat-scroll">
        {messages.length === 0 && !thinking ? (
          <Welcome onSuggestion={onSuggestion} />
        ) : (
          <>
            <div className="message-list">
              {messages.map((message) => (
                <Message key={message.id} message={message} />
              ))}
            </div>
            {thinking && (
              <div className="thinking-indicator">
                <div className="thinking-dots">
                  <span />
                  <span />
                  <span />
                </div>
                <span>Thinking…</span>
              </div>
            )}
          </>
        )}
        {tools.length > 0 && (
          <div className="tool-list">
            {tools.slice(-4).map((tool) => (
              <ToolCard key={tool.id} tool={tool} />
            ))}
          </div>
        )}
        <div ref={endRef} />
      </div>
      <div className="chat-bottom">
        <Inspector open={inspectorOpen} onToggle={onToggleInspector} execution={execution} />
        <div className="composer-wrap">
          <textarea
            aria-label="Message Data Copilot"
            value={draft}
            onChange={(event) => onDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                onSend();
              }
            }}
            placeholder={connected ? 'Ask anything about your data…' : 'Connect a gateway to start asking…'}
            rows={1}
          />
          <div className="composer-actions">
            <button className="attach-button" aria-label="Attach files" onClick={() => attachRef.current?.click()}>
              <Paperclip size={16} />
            </button>
            <input
              ref={attachRef}
              type="file"
              hidden
              multiple
              accept=".csv,.xlsx,.xls,.parquet,.json"
              onChange={(event) => {
                const files = Array.from(event.target.files ?? []);
                if (files.length) onUpload(files);
                event.target.value = '';
              }}
            />
            <span className="composer-hint">↵ to send · shift ↵ for new line</span>
            {turnActive ? (
              <button className="stop-button" aria-label="Stop run" onClick={onStop}>
                <Square size={14} fill="currentColor" />
              </button>
            ) : (
              <button className="send-button" aria-label="Send message" onClick={onSend} disabled={!draft.trim()}>
                <ArrowUp size={17} />
              </button>
            )}
          </div>
        </div>
        <div className="connection-line">
          <span className={`status-dot ${connected ? 'connected' : 'offline'}`} />{' '}
          {connected ? 'Connected to analysis agent' : 'Gateway not connected'}
          <span className="connection-latency">· secure session</span>
        </div>
      </div>
    </section>
  );
}

function Welcome({ onSuggestion }: { onSuggestion: (value: string) => void }) {
  return (
    <div className="welcome">
      <div className="welcome-icon">
        <Sparkles size={25} />
      </div>
      <span className="eyebrow">Your analytical partner</span>
      <h2>
        Start with a question.
        <br />
        <span>Leave with clarity.</span>
      </h2>
      <p>Upload a dataset and I’ll help you explore patterns, build visualizations, and turn findings into a polished deliverable.</p>
      <div className="suggestion-grid">
        {suggestions.map(({ label, icon: Icon }) => (
          <button key={label} onClick={() => onSuggestion(label)}>
            <Icon size={15} />
            {label}
            <ArrowUp size={14} className="suggestion-arrow" />
          </button>
        ))}
      </div>
      <div className="welcome-note">
        <Gauge size={14} /> Runs in an isolated, private sandbox
      </div>
    </div>
  );
}

function Message({ message }: { message: WorkbenchState['messages'][number] }) {
  const isAssistant = message.role === 'assistant';
  return (
    <div className={`message ${message.role}`}>
      {isAssistant && (
        <div className="message-avatar">
          <Sparkles size={13} />
        </div>
      )}
      <div className="message-content">
        <div className="message-meta">
          {message.role === 'user' ? 'You' : message.role === 'system' ? 'System' : 'Data Copilot'}
          <span>{new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div className={`message-text ${isAssistant ? 'markdown-body' : ''}`}>
          {isAssistant ? <ReactMarkdown>{message.content}</ReactMarkdown> : message.content}
          {message.streaming && <span className="stream-caret" />}
        </div>
      </div>
    </div>
  );
}

function ToolCard({ tool }: { tool: WorkbenchState['tools'][number] }) {
  return (
    <div className="tool-card">
      <div className={`tool-status ${tool.status}`}>
        {tool.status === 'running' ? (
          <RefreshCw size={13} className="spin" />
        ) : tool.status === 'error' ? (
          <AlertCircle size={13} />
        ) : (
          <Check size={13} />
        )}
      </div>
      <div>
        <strong>{tool.name}</strong>
        <span>{tool.status === 'running' ? 'Running in sandbox…' : tool.detail || 'Completed successfully'}</span>
      </div>
      <Code2 size={15} className="tool-code-icon" />
    </div>
  );
}

function Inspector({
  open,
  onToggle,
  execution,
}: {
  open: boolean;
  onToggle: () => void;
  execution: WorkbenchState['execution'];
}) {
  return (
    <div className={`inspector ${open ? 'inspector-open' : ''}`}>
      <button className="inspector-toggle" onClick={onToggle} aria-expanded={open}>
        <div className="inspector-title">
          <Terminal size={15} />
          <span>Execution inspector</span>
          {execution.running && (
            <span className="running-label">
              <span className="status-dot running" /> running
            </span>
          )}
        </div>
        {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
      </button>
      {open && (
        <div className="inspector-content">
          <div className="inspector-pane">
            <div className="pane-label">
              Generated code <Code2 size={12} />
            </div>
            <pre>{execution.code || '# Code will appear here when the agent runs.'}</pre>
          </div>
          <div className="inspector-pane output-pane">
            <div className="pane-label">
              Output <span className="output-tabs">stdout <i>stderr</i></span>
            </div>
            <pre className={execution.stderr ? 'has-error' : ''}>
              {execution.stderr || execution.stdout || 'No output yet.'}
            </pre>
          </div>
          <div className="inspector-pane variables-pane">
            <div className="pane-label">
              Variables <span className="count-badge">{execution.variables.length}</span>
            </div>
            {execution.variables.length === 0 ? (
              <span className="muted">No active variables</span>
            ) : (
              execution.variables.map((variable) => (
                <div className="variable-row" key={variable.name}>
                  <b>{variable.name}</b>
                  <span>{variable.type}</span>
                  <em>{variable.value}</em>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Canvas({
  sessionId,
  artifact,
  artifacts,
  tab,
  onTab,
  fullscreen,
  onFullscreen,
  onSelectArtifact,
}: {
  sessionId?: string;
  artifact?: WorkbenchState['artifact'];
  artifacts: WorkbenchState['artifacts'];
  tab: WorkbenchState['canvasTab'];
  onTab: (value: WorkbenchState['canvasTab']) => void;
  fullscreen: boolean;
  onFullscreen: () => void;
  onSelectArtifact: (name: string) => void;
}) {
  const [zoom, setZoom] = useState(100);
  const [refreshKey, setRefreshKey] = useState(0);
  const previewUrl = useMemo(() => {
    if (!sessionId || !artifact?.port) return '';
    const artifactName = artifact.name || '';
    const artifactPath = artifact.path || (/\.(html?|xhtml)$/i.test(artifactName) ? artifactName : '');
    const previewPath = artifactPath
      ? artifactPath.split('/').map((segment) => encodeURIComponent(segment)).join('/')
      : '';
    return apiUrl(`/api/sessions/${sessionId}/preview/${artifact.port}/${previewPath}`);
  }, [sessionId, artifact]);
  const pdfUrl =
    artifact?.url ||
    (sessionId && artifact?.path
      ? apiUrl(`/api/sessions/${sessionId}/artifacts/${encodeURIComponent(artifact.path)}`)
      : '');
  return (
    <section className="canvas">
      <div className="canvas-topbar">
        <div className="canvas-tabs">
          <button className={tab === 'web' ? 'active' : ''} onClick={() => onTab('web')}>
            <Globe2 size={14} /> Web app
          </button>
          <button className={tab === 'document' ? 'active' : ''} onClick={() => onTab('document')}>
            <FileText size={14} /> Document
          </button>
          <button className={tab === 'empty' ? 'active' : ''} onClick={() => onTab('empty')}>
            <PanelRight size={14} /> Empty
          </button>
        </div>
        <div className="canvas-tools">
          {artifacts.length > 1 && (
            <select
              className="artifact-select"
              aria-label="Select artifact"
              value={artifact?.name ?? ''}
              onChange={(event) => onSelectArtifact(event.target.value)}
            >
              {artifacts.map((item) => (
                <option key={item.name ?? item.path ?? item.url} value={item.name ?? ''}>
                  {item.title || item.name || 'Artifact'}
                </option>
              ))}
            </select>
          )}
          <span className="canvas-label">{artifact?.title || artifact?.name || 'Artifact canvas'}</span>
          <button className="icon-button" aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(50, value - 10))}>
            <ZoomOut size={15} />
          </button>
          <span className="zoom-label">{zoom}%</span>
          <button className="icon-button" aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(150, value + 10))}>
            <ZoomIn size={15} />
          </button>
          <span className="toolbar-divider" />
          <button className="icon-button" aria-label="Refresh artifact" onClick={() => setRefreshKey((value) => value + 1)}>
            <RefreshCw size={15} />
          </button>
          <button className="icon-button" aria-label={fullscreen ? 'Exit full screen' : 'Enter full screen'} onClick={onFullscreen}>
            {fullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
          </button>
        </div>
      </div>
      <div className="canvas-body">
        {tab === 'web' && previewUrl ? (
          <div className="preview-frame" style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'center top', width: `${10000 / zoom}%` }}>
            <iframe key={refreshKey} title="Generated web application" src={previewUrl} sandbox="allow-scripts allow-same-origin allow-forms" />
          </div>
        ) : tab === 'document' && pdfUrl ? (
          <div className="document-preview" style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top center' }}>
            <div className="pdf-sheet">
              <div className="pdf-accent" />
              <div className="pdf-kicker">DATA COPILOT · GENERATED DOCUMENT</div>
              <h2>{artifact?.title || artifact?.name || 'Analysis report'}</h2>
              <p className="pdf-placeholder">Your generated document is ready to view.</p>
              <iframe title="Generated PDF document" src={pdfUrl} />
            </div>
          </div>
        ) : (
          <CanvasPlaceholder tab={tab} hasArtifact={Boolean(artifact)} onWeb={() => onTab(artifact?.type === 'pdf' ? 'document' : 'web')} />
        )}
      </div>
      <div className="canvas-footer">
        <span>
          <span className="status-dot connected" /> Canvas ready
        </span>
        <span className="canvas-footer-right">
          {artifact ? (
            <>
              <span>{artifact.type === 'pdf' ? 'PDF artifact' : 'Interactive preview'}</span>
              <button
                onClick={() => {
                  if (pdfUrl) window.open(pdfUrl, '_blank', 'noopener,noreferrer');
                }}
              >
                <ExternalLink size={13} /> Open
              </button>
              <button
                onClick={() => {
                  if (pdfUrl) {
                    const link = document.createElement('a');
                    link.href = pdfUrl;
                    link.download = artifact.name || 'data-copilot-artifact';
                    link.click();
                  }
                }}
              >
                <Download size={13} /> Download
              </button>
            </>
          ) : (
            'Artifacts appear here after a run'
          )}
        </span>
      </div>
    </section>
  );
}

function CanvasPlaceholder({ tab, hasArtifact, onWeb }: { tab: string; hasArtifact: boolean; onWeb: () => void }) {
  return (
    <div className="canvas-placeholder">
      <div className="placeholder-grid" />
      <div className="placeholder-icon">{tab === 'document' ? <FileText size={24} /> : <Globe2 size={24} />}</div>
      <h2>{hasArtifact ? 'Choose an artifact view' : 'Your canvas is ready'}</h2>
      <p>
        {hasArtifact
          ? 'Switch to the generated artifact tab to view the result.'
          : 'Ask Copilot to create a dashboard or report. Your result will render here.'}
      </p>
      {!hasArtifact && (
        <div className="placeholder-hint">
          <Play size={13} /> Try “Build a dashboard”
        </div>
      )}
      {hasArtifact && (
        <button className="text-button" onClick={onWeb}>
          View artifact <ChevronRight size={14} />
        </button>
      )}
    </div>
  );
}
