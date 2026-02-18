/* ============================================================================
 * REST API client with admin key authentication
 * ============================================================================ */

let adminKey = '';

export function setAdminKey(key: string) {
  adminKey = key;
}

export function getAdminKey(): string {
  return adminKey;
}

function authHeaders(): Record<string, string> {
  return {
    Authorization: `Bearer ${adminKey}`,
    'Content-Type': 'application/json',
  };
}

/** Generic fetch wrapper with auth headers and error handling. */
export async function apiFetch<T = unknown>(
  url: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(url, {
    ...options,
    headers: {
      ...authHeaders(),
      ...(options.headers as Record<string, string> | undefined),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return res.json();
  }
  return (await res.text()) as unknown as T;
}

/** Validate admin key by calling /admin/stats. */
export async function validateKey(key: string): Promise<boolean> {
  const res = await fetch('/admin/stats', {
    headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
  });
  return res.ok;
}
