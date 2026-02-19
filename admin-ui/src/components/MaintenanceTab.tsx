/* ============================================================================
 * Maintenance Tab
 *
 * Sections:
 *   - Embedding Info (read-only model status)
 *   - Re-embedding Controls (blue-green migration with progress)
 *   - Template Learning (boilerplate detection)
 * ============================================================================ */

import { useCallback, useEffect, useRef, useState } from 'react';
import { marked } from 'marked';
import { apiFetch } from '../api/client';
import { useApp } from '../store';
import type { MaintenanceTask, ModelConfig, TemplateDomain, TemplateInfo } from '../types';

export default function MaintenanceTab() {
  return (
    <div className="bg-white rounded-lg shadow-md p-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">⚙️ System Maintenance</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ReembedSection />
        <div className="space-y-6">
          <EmbeddingInfoCard />
        </div>
      </div>
      <TemplateLearningSection />
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Re-embedding Section                                                       */
/* -------------------------------------------------------------------------- */

function ReembedSection() {
  const { collections } = useApp();
  const [selectedCollection, setSelectedCollection] = useState('');
  const [vectorTypes, setVectorTypes] = useState({ dense: true, colbert: true });
  const [status, setStatus] = useState('');
  const [tasks, setTasks] = useState<Record<string, MaintenanceTask>>({});
  const [polling, setPolling] = useState(false);
  const intervalRef = useRef<number | null>(null);

  // Filter: main websearch collection + kv_* + *_faq
  const reembedCollections = collections.filter((c) => {
    const name = c.alias || c.name;
    if (name.startsWith('kv_')) return true;
    if (name.endsWith('_faq')) return true;
    // Main document collection (not migration leftovers, not feedback, not system)
    if (c.type === 'documents' && !name.includes('_migration_') && !name.endsWith('_feedback') && !name.startsWith('__') && !name.startsWith('open_webui') && name !== 'system_config') return true;
    return false;
  });

  // Auto-select first collection if none selected
  useEffect(() => {
    if (!selectedCollection && reembedCollections.length > 0) {
      const main = reembedCollections.find((c) => c.type === 'documents');
      setSelectedCollection((main || reembedCollections[0]).alias || (main || reembedCollections[0]).name);
    }
  }, [reembedCollections, selectedCollection]);

  const toggleType = (t: 'dense' | 'colbert') =>
    setVectorTypes((v) => ({ ...v, [t]: !v[t] }));

  const startReembedding = async () => {
    if (!selectedCollection) { alert('Please select a collection.'); return; }
    const types = Object.entries(vectorTypes)
      .filter(([, v]) => v)
      .map(([k]) => k);
    if (types.length === 0) {
      alert('Please select at least one vector type to re-embed.');
      return;
    }

    if (!confirm(`Are you sure you want to re-embed vectors in "${selectedCollection}"? This cannot be stopped once started.`)) return;

    setStatus('Starting…');
    try {
      const body: Record<string, unknown> = { vector_types: types, collection_name: selectedCollection };
      const res = await apiFetch<{ status: string; message: string }>('/admin/maintenance/re-embed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      setStatus(`✅ ${res.message}`);
      startPolling();
    } catch (e) {
      setStatus('❌ Error: ' + (e instanceof Error ? e.message : e));
    }
  };

  const pollStatus = useCallback(async () => {
    try {
      const data = await apiFetch<Record<string, MaintenanceTask>>('/admin/maintenance/status');
      setTasks(data);
      const anyActive = Object.values(data).some((t) => t.status === 'in-progress');
      if (!anyActive && Object.keys(data).length > 0) {
        stopPolling();
      }
    } catch (e) {
      console.error('Failed to poll status:', e);
    }
  }, []);

  const startPolling = useCallback(() => {
    if (intervalRef.current) return;
    setPolling(true);
    pollStatus();
    intervalRef.current = window.setInterval(pollStatus, 3000);
  }, [pollStatus]);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setPolling(false);
  }, []);

  useEffect(() => {
    // Check if there are any active tasks on mount
    pollStatus();
    return () => stopPolling();
  }, [pollStatus, stopPolling]);

  const finalizeMigration = async (collectionName: string) => {
    if (!confirm(`Finalize migration for "${collectionName}"? This will swap the alias to the new collection.`)) return;
    try {
      const res = await apiFetch<{ status: string }>('/admin/maintenance/finalize-migration', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ collection_name: collectionName, delete_old: true }),
      });
      alert(`Migration finalized: ${res.status}`);
      pollStatus();
    } catch (e) {
      alert('Error: ' + (e instanceof Error ? e.message : e));
    }
  };

  const displayName = (c: { name: string; alias?: string; count: number }) => {
    if (c.alias) return `${c.alias} (${c.count} pts)`;
    return `${c.name} (${c.count} pts)`;
  };

  return (
    <div className="border rounded-lg p-6 bg-blue-50 border-blue-200">
      <h3 className="text-lg font-bold text-blue-800 mb-2">🔄 Re-embed Vectors</h3>
      <p className="text-sm text-blue-700 mb-4">
        Regenerate embeddings for a collection. Select the target collection and which vector types to re-compute.
      </p>

      <div className="mb-4">
        <label className="block text-xs font-bold text-blue-800 uppercase mb-1">Collection</label>
        <select
          value={selectedCollection}
          onChange={(e) => setSelectedCollection(e.target.value)}
          className="w-full border border-blue-300 rounded px-3 py-2 text-sm bg-white focus:ring-2 focus:ring-blue-400"
        >
          <option value="">— Select collection —</option>
          {reembedCollections.map((c) => (
            <option key={c.name} value={c.alias || c.name}>
              {displayName(c)}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-4 space-y-2">
        <label className="block text-xs font-bold text-blue-800 uppercase">Vector Types to Re-embed:</label>
        <div className="flex gap-4">
          {(['dense', 'colbert'] as const).map((t) => (
            <label key={t} className="flex items-center gap-2 cursor-pointer text-sm">
              <input type="checkbox" checked={vectorTypes[t]} onChange={() => toggleType(t)} className="h-4 w-4 text-blue-600" />
              <span>{t === 'dense' ? 'Dense Vectors' : 'ColBERT Vectors'}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <button onClick={startReembedding} className="btn-primary" disabled={!selectedCollection}>
          Re-embed Collection
        </button>
        {status && <div className="text-sm font-medium text-gray-600 mt-2">{status}</div>}

        {/* Progress display */}
        {Object.entries(tasks).map(([key, task]) => (
          <TaskProgress key={key} taskKey={key} task={task} onFinalize={finalizeMigration} />
        ))}
      </div>

      {!polling && Object.keys(tasks).length > 0 && (
        <button onClick={startPolling} className="btn-secondary text-xs mt-3">↻ Refresh Status</button>
      )}
    </div>
  );
}

function TaskProgress({
  taskKey,
  task,
  onFinalize,
}: {
  taskKey: string;
  task: MaintenanceTask;
  onFinalize: (name: string) => void;
}) {
  const pct = task.total > 0 ? Math.round((task.completed / task.total) * 100) : 0;

  const bgColor =
    task.status === 'finalized'
      ? 'bg-green-50 border-green-200'
      : task.status === 'failed'
        ? 'bg-red-50 border-red-200'
        : task.status === 'awaiting_finalize'
          ? 'bg-yellow-50 border-yellow-200'
          : 'bg-white border-blue-100';

  return (
    <div className={`mt-4 p-4 rounded border shadow-sm ${bgColor}`}>
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-blue-600 bg-blue-100">
          {taskKey}
        </span>
        <span className="text-xs font-semibold inline-block py-1 px-2 text-blue-600">
          {task.status === 'in-progress' ? `${pct}%` : task.status}
        </span>
      </div>

      {task.status === 'in-progress' && (
        <div className="overflow-hidden h-3 mb-2 text-xs flex rounded bg-blue-100 border border-blue-200">
          <div style={{ width: `${pct}%` }} className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500 transition-all duration-500" />
        </div>
      )}

      <div className="text-xs text-gray-500 font-mono">
        {task.completed}/{task.total} points • Vectors: {task.vector_types?.join(', ') || 'all'}
        {task.error && <span className="text-red-600 ml-2">Error: {task.error}</span>}
      </div>

      {task.status === 'awaiting_finalize' && (
        <button
          onClick={() => onFinalize(task.alias_name || taskKey.split(':')[0])}
          className="btn-primary text-sm mt-2"
          style={{ backgroundColor: '#16a34a' }}
        >
          ✅ Finalize Migration
        </button>
      )}
    </div>
  );
}

function EmbeddingInfoCard() {
  const [config, setConfig] = useState<ModelConfig | null>(null);

  useEffect(() => {
    apiFetch<ModelConfig>('/admin/maintenance/config/models')
      .then(setConfig)
      .catch(() => {});
  }, []);

  return (
    <div className="border rounded-lg p-6 bg-gray-50">
      <h3 className="text-lg font-bold text-gray-800 mb-2">ℹ️ Embedding Models</h3>
      <ul className="text-sm text-gray-600 space-y-2 list-disc pl-5">
        <li><strong>ColBERT:</strong> <code className="text-xs bg-gray-200 px-1 rounded">{config?.colbert_model_id || '…'}</code> (128-dim)</li>
        <li><strong>Dense:</strong> <code className="text-xs bg-gray-200 px-1 rounded">{config?.dense_model_id || '…'}</code> ({config?.dense_vector_size || '…'}-dim, auto-detected)</li>
      </ul>
      <p className="text-xs text-gray-400 mt-4">
        Models are configured via environment variables (<code>DENSE_MODEL_NAME</code>, <code>COLBERT_MODEL_NAME</code>, <code>DENSE_EMBEDDING_URL</code>).
        The dense vector dimension is auto-detected from the OpenAI-compatible endpoint at startup.
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Template Learning Section                                                  */
/* -------------------------------------------------------------------------- */

function TemplateLearningSection() {
  const { currentCollection } = useApp();
  const [domains, setDomains] = useState<TemplateDomain[]>([]);
  const [selectedDomain, setSelectedDomain] = useState('');
  const [threshold, setThreshold] = useState(0.5);
  const [minPages, setMinPages] = useState(5);
  const [scrollLimit, setScrollLimit] = useState(2000);
  const [status, setStatus] = useState('');
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [preview, setPreview] = useState<PreviewResult | null>(null);

  const loadDomains = useCallback(async () => {
    try {
      const params = currentCollection ? `?collection_name=${encodeURIComponent(currentCollection)}` : '';
      const data = await apiFetch<{ domains: TemplateDomain[] }>(`/admin/templates/domains${params}`);
      setDomains(data.domains || []);
    } catch (e) {
      console.error('Failed to load domains:', e);
    }
  }, [currentCollection]);

  const loadTemplates = useCallback(async () => {
    try {
      const params = currentCollection ? `?collection_name=${encodeURIComponent(currentCollection)}` : '';
      const data = await apiFetch<{ templates: TemplateInfo[] }>(`/admin/templates${params}`);
      setTemplates(data.templates || []);
    } catch (e) {
      console.error('Failed to load templates:', e);
    }
  }, [currentCollection]);

  useEffect(() => {
    loadDomains();
    loadTemplates();
  }, [loadDomains, loadTemplates]);

  const runPreview = async () => {
    if (!selectedDomain) { alert('Select a domain first.'); return; }
    setStatus('Running preview…');
    setPreview(null);
    try {
      const params = new URLSearchParams({
        domain: selectedDomain,
        threshold: threshold.toString(),
        min_pages: minPages.toString(),
        scroll_limit: scrollLimit.toString(),
        sample_count: '3',
      });
      if (currentCollection) params.set('collection_name', currentCollection);
      const data = await apiFetch<PreviewResult>(`/admin/templates/preview?${params}`);
      setPreview(data);
      setStatus(`✅ Preview completed — ${data.boilerplate_fingerprints?.length || 0} boilerplate blocks detected`);
    } catch (e) {
      setStatus('❌ Error: ' + (e instanceof Error ? e.message : e));
    }
  };

  const runBuild = async () => {
    if (!selectedDomain) { alert('Select a domain first.'); return; }
    if (!confirm(`Build template for "${selectedDomain}"? This will affect future scrapes.`)) return;
    setStatus('Building template…');
    try {
      const params = new URLSearchParams({
        domain: selectedDomain,
        threshold: threshold.toString(),
        min_pages: minPages.toString(),
        scroll_limit: scrollLimit.toString(),
      });
      if (currentCollection) params.set('collection_name', currentCollection);
      await apiFetch(`/admin/templates/build?${params}`, { method: 'POST' });
      setStatus('✅ Template built successfully');
      loadTemplates();
    } catch (e) {
      setStatus('❌ Error: ' + (e instanceof Error ? e.message : e));
    }
  };

  const deleteTemplate = async (domain: string) => {
    if (!confirm(`Delete template for "${domain}"?`)) return;
    try {
      const params = currentCollection ? `?collection_name=${encodeURIComponent(currentCollection)}` : '';
      await apiFetch(`/admin/templates/${encodeURIComponent(domain)}${params}`, { method: 'DELETE' });
      loadTemplates();
    } catch (e) {
      alert('Error: ' + (e instanceof Error ? e.message : e));
    }
  };

  const reapplyTemplate = async (domain: string) => {
    if (!confirm(`Reapply template for "${domain}" to all existing documents?\nThis will re-filter content and re-embed in the background.`)) return;
    setStatus(`♻️ Reapplying template for ${domain}…`);
    try {
      const params = new URLSearchParams({ domain });
      if (currentCollection) params.set('collection_name', currentCollection);
      const data = await apiFetch<{ status: string; message: string }>(`/admin/templates/reapply?${params}`, { method: 'POST' });
      setStatus(`✅ ${data.message}`);
    } catch (e) {
      setStatus('❌ Reapply error: ' + (e instanceof Error ? e.message : e));
    }
  };

  return (
    <div className="mt-8 border rounded-lg p-6 bg-amber-50 border-amber-200">
      <h3 className="text-lg font-bold text-amber-800 mb-2">🧹 Template Learning (Boilerplate Detection)</h3>
      <p className="text-sm text-amber-700 mb-4">
        Analyse pages from a domain to identify repeating boilerplate blocks (navigation, footers, cookie banners).
        Preview the results before committing — blocks appearing on ≥ threshold% of pages will be filtered from future scrapes.
      </p>

      {/* Domain & params */}
      <div className="flex flex-wrap items-end gap-3 mb-4">
        <div className="flex-1 min-w-[200px]">
          <label className="block text-xs font-semibold text-amber-800 mb-1">Domain</label>
          <div className="flex gap-2">
            <select
              value={selectedDomain}
              onChange={(e) => setSelectedDomain(e.target.value)}
              className="flex-1 border border-amber-300 rounded px-3 py-2 text-sm bg-white focus:ring-2 focus:ring-amber-400"
            >
              <option value="">— Select domain —</option>
              {domains.map((d) => (
                <option key={d.domain} value={d.domain}>
                  {d.domain} ({d.page_count} pages)
                </option>
              ))}
            </select>
            <button onClick={loadDomains} className="btn-secondary text-xs px-3" title="Refresh domain list">↻</button>
          </div>
        </div>
        <div className="w-28">
          <label className="block text-xs font-semibold text-amber-800 mb-1">Threshold</label>
          <input type="number" value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} step={0.05} min={0.1} max={1.0}
            className="w-full border border-amber-300 rounded px-3 py-2 text-sm bg-white focus:ring-2 focus:ring-amber-400" />
        </div>
        <div className="w-28">
          <label className="block text-xs font-semibold text-amber-800 mb-1">Min Pages</label>
          <input type="number" value={minPages} onChange={(e) => setMinPages(parseInt(e.target.value))} min={2} max={100}
            className="w-full border border-amber-300 rounded px-3 py-2 text-sm bg-white focus:ring-2 focus:ring-amber-400" />
        </div>
        <div className="w-28">
          <label className="block text-xs font-semibold text-amber-800 mb-1">Max Scan</label>
          <input type="number" value={scrollLimit} onChange={(e) => setScrollLimit(parseInt(e.target.value))} min={100} max={50000} step={500}
            className="w-full border border-amber-300 rounded px-3 py-2 text-sm bg-white focus:ring-2 focus:ring-amber-400"
            title="Maximum pages to scan for this domain" />
        </div>
        <div className="flex gap-2">
          <button onClick={runPreview} className="btn-secondary">👁 Preview</button>
          <button onClick={runBuild} className="btn-primary">▶ Build Template</button>
        </div>
      </div>

      {status && <div className="text-sm font-medium text-gray-600 mb-3">{status}</div>}

      {/* Existing templates */}
      {templates.length > 0 && (
        <div className="mb-4">
          <h4 className="text-sm font-bold text-amber-800 mb-2">📋 Existing Templates</h4>
          <div className="text-xs space-y-1">
            {templates.map((t) => (
              <div key={t.domain} className="flex items-center justify-between bg-white rounded px-3 py-2 border border-amber-200">
                <span>
                  <strong>{t.domain}</strong> — {t.page_count} pages, {t.fingerprint_count} fingerprints, threshold: {t.threshold}
                </span>
                <div className="flex gap-2">
                  <button onClick={() => reapplyTemplate(t.domain)} className="text-blue-600 hover:text-blue-800 text-xs" title="Reapply template to all existing documents">♻️ Reapply</button>
                  <button onClick={() => deleteTemplate(t.domain)} className="text-red-500 hover:text-red-700 text-xs">🗑 Delete</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Preview results */}
      {preview && <TemplatePreviewDisplay preview={preview} />}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Template Preview Display                                                   */
/* -------------------------------------------------------------------------- */

interface PreviewResult {
  boilerplate_fingerprints?: string[];
  samples?: Array<{
    url: string;
    before_length: number;
    after_length: number;
    before_content?: string;
    after_content?: string;
  }>;
  pages_analysed?: number;
  domain?: string;
}

function TemplatePreviewDisplay({ preview }: { preview: PreviewResult }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-bold text-amber-800">Preview Results</h4>
        <span className="text-xs text-amber-700 bg-amber-100 px-2 py-1 rounded">
          {preview.pages_analysed || 0} pages analysed — {preview.boilerplate_fingerprints?.length || 0} blocks detected
        </span>
      </div>

      {/* Detected boilerplate blocks */}
      {preview.boilerplate_fingerprints && preview.boilerplate_fingerprints.length > 0 && (
        <div className="mb-4">
          <h5 className="text-xs font-semibold text-gray-700 mb-1">Detected boilerplate blocks:</h5>
          <div className="max-h-40 overflow-y-auto bg-white border border-amber-200 rounded p-2 text-xs font-mono space-y-1">
            {preview.boilerplate_fingerprints.map((fp, i) => (
              <div key={i} className="p-1 bg-amber-50 rounded">{fp}</div>
            ))}
          </div>
        </div>
      )}

      {/* Sample before/after */}
      {preview.samples && preview.samples.length > 0 && (
        <div className="space-y-4">
          {preview.samples.map((s, i) => (
            <div key={i} className="border rounded bg-white p-4">
              <div className="text-sm font-medium text-gray-800 mb-2">
                <a href={s.url} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">{s.url}</a>
              </div>
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <div className="font-semibold text-gray-600 mb-1">Before ({s.before_length} chars)</div>
                  {s.before_content ? (
                    <div className="max-h-48 overflow-auto bg-red-50 p-2 rounded markdown-content" dangerouslySetInnerHTML={{ __html: marked.parse(s.before_content.substring(0, 2000)) as string }} />
                  ) : (
                    <div className="bg-gray-50 p-2 rounded text-gray-400">No content</div>
                  )}
                </div>
                <div>
                  <div className="font-semibold text-gray-600 mb-1">After ({s.after_length} chars)</div>
                  {s.after_content ? (
                    <div className="max-h-48 overflow-auto bg-green-50 p-2 rounded markdown-content" dangerouslySetInnerHTML={{ __html: marked.parse(s.after_content.substring(0, 2000)) as string }} />
                  ) : (
                    <div className="bg-gray-50 p-2 rounded text-gray-400">No content</div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
