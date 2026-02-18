/* ============================================================================
 * MCP Client — Streamable HTTP transport
 *
 * Calls MCP tools directly via the /mcp-server/mcp endpoint.
 * Handles session initialization, SSE responses, and result normalization.
 * ============================================================================ */

export class MCPClient {
  private endpoint: string;
  private messageId = 0;
  private sessionId: string | null = null;
  private protocolVersion: string | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(endpoint = '/mcp-server/mcp') {
    this.endpoint = endpoint;
  }

  // --- Session lifecycle ---

  private async ensureSession(): Promise<void> {
    if (this.sessionId) return;
    if (!this.initPromise) {
      this.initPromise = this._initializeSession();
    }
    await this.initPromise;
  }

  private async _initializeSession(): Promise<void> {
    try {
      const id = ++this.messageId;
      const res = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'initialize',
          params: {
            protocolVersion: '2025-11-25',
            capabilities: {},
            clientInfo: { name: 'qdrant-proxy-admin', version: '1.0.0' },
          },
          id,
        }),
      });
      if (!res.ok) throw new Error(`MCP initialize failed: ${res.status}`);
      const result = await this.parseResponse(res);
      this.sessionId = res.headers.get('mcp-session-id');
      this.protocolVersion = (result as Record<string, unknown>)?.protocolVersion as string || '2025-11-25';
      if (!this.sessionId) throw new Error('MCP initialize returned no session ID');
      await this.sendInitialized();
    } catch (e) {
      this.initPromise = null;
      throw e;
    }
  }

  private async sendInitialized(): Promise<void> {
    await fetch(this.endpoint, {
      method: 'POST',
      headers: this.headers('application/json, text/event-stream'),
      body: JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized', params: null }),
    });
  }

  // --- Headers ---

  private headers(accept: string): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json', Accept: accept };
    if (this.sessionId) h['mcp-session-id'] = this.sessionId;
    if (this.protocolVersion) h['mcp-protocol-version'] = this.protocolVersion;
    return h;
  }

  // --- Response parsing ---

  private async parseResponse(res: Response): Promise<unknown> {
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('text/event-stream')) return this.handleSSE(res);

    const raw = await res.text();
    if (!raw) return null;
    try {
      const parsed = JSON.parse(raw);
      if (parsed.error) throw new Error(parsed.error.message || JSON.stringify(parsed.error));
      return parsed.result;
    } catch {
      if (raw.includes('event:')) return this.parseSSEText(raw);
      throw new Error('Failed to parse MCP response');
    }
  }

  private parseSSEText(text: string): unknown {
    let result: unknown = null;
    for (const line of text.split('\n')) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') continue;
      const parsed = JSON.parse(data);
      if (parsed.result) result = parsed.result;
      else if (parsed.error) throw new Error(parsed.error.message || JSON.stringify(parsed.error));
    }
    return result;
  }

  private async handleSSE(res: Response): Promise<unknown> {
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let result: unknown = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') continue;
        try {
          const parsed = JSON.parse(data);
          if (parsed.result) result = parsed.result;
          else if (parsed.error) throw new Error(parsed.error.message);
        } catch { /* skip non-JSON */ }
      }
    }
    return result;
  }

  // --- Result normalization ---

  private normalize(result: unknown): unknown {
    if (!result || typeof result !== 'object') return result;
    const r = result as Record<string, unknown>;
    if (r.structuredContent) return r.structuredContent;
    if (Array.isArray(r.content)) {
      for (const item of r.content as Array<Record<string, unknown>>) {
        if (item?.type === 'text' && typeof item.text === 'string') {
          try { return JSON.parse(item.text as string); } catch { /* keep scanning */ }
        }
      }
    }
    return result;
  }

  // --- Public API ---

  async callTool<T = unknown>(toolName: string, args: Record<string, unknown> = {}): Promise<T> {
    await this.ensureSession();
    const id = ++this.messageId;
    const res = await fetch(this.endpoint, {
      method: 'POST',
      headers: this.headers('application/json, text/event-stream'),
      body: JSON.stringify({ jsonrpc: '2.0', method: 'tools/call', params: { name: toolName, arguments: args }, id }),
    });
    if (!res.ok) throw new Error(`MCP request failed: ${res.status}`);
    const result = await this.parseResponse(res);
    return this.normalize(result) as T;
  }
}

/** Global singleton MCP client */
export const mcpClient = new MCPClient();
