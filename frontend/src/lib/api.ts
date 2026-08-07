const configuredApi = import.meta.env.VITE_API_URL?.replace(/\/$/, '');

export const apiBase = configuredApi || '';

export function apiUrl(path: string) {
  return `${apiBase}${path.startsWith('/') ? path : `/${path}`}`;
}

export function websocketUrl(path: string) {
  if (import.meta.env.VITE_WS_URL) {
    return `${import.meta.env.VITE_WS_URL.replace(/\/$/, '')}${path.startsWith('/') ? path : `/${path}`}`;
  }
  if (configuredApi) {
    const base = new URL(configuredApi);
    base.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
    base.pathname = `${base.pathname.replace(/\/$/, '')}${path.startsWith('/') ? path : `/${path}`}`;
    return base.toString();
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path.startsWith('/') ? path : `/${path}`}`;
}

export async function createSession() {
  const response = await fetch(apiUrl('/api/sessions'), { method: 'POST' });
  if (!response.ok) throw new Error(`Session could not be created (${response.status})`);
  const data = (await response.json()) as Record<string, unknown>;
  const id = data.id ?? data.session_id ?? data.sessionId;
  if (!id) throw new Error('The gateway did not return a session id');
  return String(id);
}

export async function uploadDataset(sessionId: string, file: File) {
  const form = new FormData();
  form.append('files', file);
  const response = await fetch(apiUrl(`/api/sessions/${sessionId}/files`), {
    method: 'POST',
    body: form,
  });
  if (!response.ok) throw new Error(`Upload failed (${response.status})`);
  return (await response.json().catch(() => ({}))) as Record<string, unknown>;
}
