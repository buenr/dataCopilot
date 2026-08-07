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

export async function getSession(sessionId: string) {
  const response = await fetch(apiUrl(`/api/sessions/${sessionId}`));
  return response;
}

export async function deleteSession(sessionId: string) {
  try {
    await fetch(apiUrl(`/api/sessions/${sessionId}`), { method: 'DELETE' });
  } catch {
    // Best-effort: a dropped gateway or already-reaped session is not fatal.
  }
}

export async function uploadDataset(sessionId: string, files: File[]) {
  const form = new FormData();
  for (const file of files) form.append('files', file);
  const response = await fetch(apiUrl(`/api/sessions/${sessionId}/files`), {
    method: 'POST',
    body: form,
  });
  if (!response.ok) throw new Error(`Upload failed (${response.status})`);
  return (await response.json().catch(() => ({}))) as Record<string, unknown>;
}

export async function deleteDataset(sessionId: string, name: string) {
  const response = await fetch(apiUrl(`/api/sessions/${sessionId}/files/${encodeURIComponent(name)}`), { method: 'DELETE' });
  if (!response.ok) throw new Error(`Dataset removal failed (${response.status})`);
  return (await response.json().catch(() => ({}))) as Record<string, unknown>;
}
