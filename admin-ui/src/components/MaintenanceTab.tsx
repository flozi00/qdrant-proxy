/* ============================================================================
 * Maintenance Tab
 *
 * Sections:
 *   - Embedding Info (read-only model status)
 *   - Re-embedding Controls (blue-green migration with progress)
 * ============================================================================ */

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch } from '../api/client';
import { useApp } from '../store';
import type { MaintenanceTask, ModelConfig } from '../types';

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
