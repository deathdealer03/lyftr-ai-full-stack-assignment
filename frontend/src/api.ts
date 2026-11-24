import { ChatMetadata, HealthStatus } from './types';

export const API_BASE_URL = (
  import.meta.env.VITE_BACKEND_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  'http://localhost:8000'
).replace(/\/$/, '');

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed (${res.status})`);
  }
  return res.json();
}

export async function sendChatMessage(
  sessionId: string,
  message: string,
  systemPrompt?: string,
  simulateFailure?: string
): Promise<{ session_id: string; reply: string; metadata: ChatMetadata }> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (simulateFailure && simulateFailure.trim()) {
    headers['X-Simulate-Failure'] = simulateFailure.trim();
  }

  const payload: { session_id: string; message: string; system_prompt?: string } = {
    session_id: sessionId,
    message: message.trim(),
  };
  if (systemPrompt && systemPrompt.trim()) {
    payload.system_prompt = systemPrompt.trim();
  }

  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    let msg = 'Failed to send message';
    if (data && data.detail) {
      msg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    } else if (res.status === 503) {
      msg = 'all providers unavailable';
    } else if (res.status === 422) {
      msg = 'Validation error (invalid session_id or message)';
    }
    throw new Error(msg);
  }

  return data;
}

export async function fetchSessionHistory(sessionId: string) {
  const res = await fetch(`${API_BASE_URL}/sessions/${sessionId}/messages`);
  if (!res.ok) {
    if (res.status === 404) return null;
    throw new Error(`Error loading session history (${res.status})`);
  }
  return res.json();
}
