import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import { apiFetch } from '../api/client';
import { isPrimarySearchCollection, useApp } from '../store';
import type {
  FAQAgentRunRequest,
  FAQAgentRunResponse,
  FAQAgentRunsResponse,
  FAQAgentRunStatus,
} from '../types';

const ACTIVE_RUN_STATUSES = new Set(['queued', 'in-progress', 'stopping']);

function statusPillClass(status: string) {
  switch (status) {
    case 'completed':
      return 'bg-green-100 text-green-800';
    case 'cancelled':
      return 'bg-gray-200 text-gray-700';
    case 'failed':
      return 'bg-red-100 text-red-700';
    case 'stopping':
      return 'bg-orange-100 text-orange-700';
    case 'in-progress':
      return 'bg-blue-100 text-blue-700';
    default:
      return 'bg-yellow-100 text-yellow-800';
  }
}

function isActiveRun(run: FAQAgentRunStatus) {
  return ACTIVE_RUN_STATUSES.has(run.status);
}

export default function FaqAgentRunsPanel() {
  const { collections, currentCollection } = useApp();
  const documentCollections = useMemo(
    () => collections.filter(isPrimarySearchCollection),
    [collections],
  );
  const [selectedCollection, setSelectedCollection] = useState(currentCollection);
  const [runs, setRuns] = useState<FAQAgentRunStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [stoppingRunId, setStoppingRunId] = useState('');
  const [message, setMessage] = useState('');
  const [form, setForm] = useState<FAQAgentRunRequest>({
    collection_name: currentCollection || undefined,
    limit_documents: 50,
    follow_links: true,
    max_hops: 1,
    max_linked_documents: 3,
    max_faqs_per_document: 3,
    force_reprocess: false,
    remove_stale_faqs: true,
  });
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (!selectedCollection && currentCollection) {
      setSelectedCollection(currentCollection);
    }
  }, [currentCollection, selectedCollection]);

  useEffect(() => {
    setForm((prev) => ({
      ...prev,
      collection_name: selectedCollection || undefined,
    }));
  }, [selectedCollection]);

  const loadRuns = useCallback(async () => {
    try {
      const data = await apiFetch<FAQAgentRunsResponse>('/admin/faq-agent/runs');
      setRuns(data.items);
      return data.items;
    } catch (error) {
      console.error('Failed to load FAQ agent runs:', error);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  const stopPolling = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const startPolling = useCallback(() => {
    if (intervalRef.current !== null) return;
    intervalRef.current = window.setInterval(async () => {
      const items = await loadRuns();
      if (!items.some(isActiveRun)) {
        stopPolling();
      }
    }, 3000);
  }, [loadRuns, stopPolling]);

  useEffect(() => {
    loadRuns().then((items) => {
      if (items.some(isActiveRun)) {
        startPolling();
      }
    });
    return () => stopPolling();
  }, [loadRuns, startPolling, stopPolling]);

  const activeRun = useMemo(
    () =>
      runs.find(
        (run) => run.collection_name === selectedCollection && isActiveRun(run),
      ) || runs.find(isActiveRun),
    [runs, selectedCollection],
  );

  const handleNumberField =
    (field: keyof FAQAgentRunRequest) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = parseInt(event.target.value, 10);
      setForm((prev) => ({ ...prev, [field]: Number.isFinite(value) ? value : 0 }));
    };

  const handleCheckboxField =
    (field: keyof FAQAgentRunRequest) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      setForm((prev) => ({ ...prev, [field]: event.target.checked }));
    };

  const handleStartRun = async () => {
    if (!selectedCollection) {
      setMessage('Select a document collection first.');
      return;
    }
    setStarting(true);
    setMessage('');
    try {
      const response = await apiFetch<FAQAgentRunResponse>('/admin/faq-agent/runs', {
        method: 'POST',
        body: JSON.stringify({
          ...form,
          collection_name: selectedCollection,
        }),
      });
      setMessage(`✅ ${response.message}`);
      const items = await loadRuns();
      if (items.some(isActiveRun)) startPolling();
    } catch (error) {
      setMessage(`❌ ${error instanceof Error ? error.message : error}`);
    } finally {
      setStarting(false);
    }
  };

  const handleStopRun = async (run: FAQAgentRunStatus) => {
    if (!confirm(`Stop FAQ run ${run.run_id} for ${run.collection_name}?`)) return;
    setStoppingRunId(run.run_id);
    setMessage('');
    try {
      const response = await apiFetch<FAQAgentRunResponse>(
        `/admin/faq-agent/runs/${run.run_id}/stop`,
        { method: 'POST' },
      );
      setMessage(`🛑 ${response.message}`);
      await loadRuns();
      startPolling();
    } catch (error) {
      setMessage(`❌ ${error instanceof Error ? error.message : error}`);
    } finally {
      setStoppingRunId('');
    }
  };

  const renderRunCard = (run: FAQAgentRunStatus) => {
    const progress =
      run.limit_documents > 0
        ? Math.min(100, Math.round((run.documents_completed / run.limit_documents) * 100))
        : 0;

    return (
      <div key={run.run_id} className="border rounded-lg p-4 bg-white shadow-sm">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <h4 className="font-semibold text-gray-800">{run.collection_name}</h4>
              <span className={`px-2 py-1 rounded-full text-xs font-semibold ${statusPillClass(run.status)}`}>
                {run.status}
              </span>
              {run.cancel_requested && run.status !== 'cancelled' && (
                <span className="px-2 py-1 rounded-full text-xs font-semibold bg-orange-100 text-orange-700">
                  cancel requested
                </span>
              )}
            </div>
            <div className="text-xs text-gray-500 break-all">
              <div>Run ID: {run.run_id}</div>
              <div>Started: {new Date(run.start_time).toLocaleString()}</div>
              {run.end_time && <div>Ended: {new Date(run.end_time).toLocaleString()}</div>}
              {run.current_document_url && <div>Current: {run.current_document_url}</div>}
            </div>
          </div>
          {isActiveRun(run) && (
            <button
              onClick={() => handleStopRun(run)}
              disabled={stoppingRunId === run.run_id}
              className="btn-danger text-sm self-start"
            >
              {stoppingRunId === run.run_id ? 'Stopping…' : 'Stop Run'}
            </button>
          )}
        </div>

        <div className="mt-4">
          <div className="flex justify-between text-xs text-gray-600 mb-1">
            <span>
              {run.documents_completed}/{run.limit_documents} documents
            </span>
            <span>{progress}%</span>
          </div>
          <div className="overflow-hidden h-2 rounded bg-gray-200">
            <div
              className="h-2 bg-blue-500 transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3 mt-4 text-sm">
          <StatCell label="Processed" value={run.documents_processed} />
          <StatCell label="Skipped" value={run.documents_skipped} />
          <StatCell label="Failed" value={run.documents_failed} />
          <StatCell label="Created" value={run.faqs_created} />
          <StatCell label="Merged" value={run.faqs_merged} />
          <StatCell label="Refreshed" value={run.faqs_refreshed} />
          <StatCell label="Reassigned" value={run.faqs_reassigned} />
          <StatCell label="Source removals" value={run.faqs_removed_sources} />
          <StatCell label="Deleted" value={run.faqs_deleted} />
          <StatCell label="Max hops" value={run.max_hops} />
          <StatCell label="Linked docs" value={run.max_linked_documents} />
          <StatCell label="FAQ/doc" value={run.max_faqs_per_document} />
        </div>

        <div className="mt-4 grid grid-cols-1 xl:grid-cols-2 gap-4">
          <div className="border rounded-md p-3 bg-gray-50">
            <div className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
              Run options
            </div>
            <div className="text-sm text-gray-700 space-y-1">
              <div>Follow links: {run.follow_links ? 'Yes' : 'No'}</div>
              <div>Force reprocess: {run.force_reprocess ? 'Yes' : 'No'}</div>
              <div>Remove stale FAQs: {run.remove_stale_faqs ? 'Yes' : 'No'}</div>
              <div>Handled documents: {run.handled_document_ids.length}</div>
            </div>
          </div>
          <div className="border rounded-md p-3 bg-gray-50">
            <div className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
              Recent documents
            </div>
            {run.recent_documents.length === 0 ? (
              <p className="text-sm text-gray-500">No documents processed yet.</p>
            ) : (
              <div className="space-y-2 max-h-56 overflow-y-auto">
                {run.recent_documents
                  .slice()
                  .reverse()
                  .map((entry) => (
                    <div key={`${run.run_id}-${entry.doc_id}-${entry.status}`} className="text-sm border-b border-gray-200 pb-2 last:border-b-0 last:pb-0">
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold ${statusPillClass(entry.status)}`}>
                          {entry.status}
                        </span>
                        <span className="font-medium text-gray-700">{entry.doc_id}</span>
                      </div>
                      {entry.url && <div className="text-xs text-gray-500 break-all">{entry.url}</div>}
                      {typeof entry.generated_faq_count === 'number' && (
                        <div className="text-xs text-gray-600">Generated FAQs: {entry.generated_faq_count}</div>
                      )}
                      {entry.error && <div className="text-xs text-red-600">{entry.error}</div>}
                    </div>
                  ))}
              </div>
            )}
          </div>
        </div>

        {run.error && <div className="mt-3 text-sm text-red-600">Error: {run.error}</div>}
      </div>
    );
  };

  return (
    <div className="border rounded-lg p-6 bg-purple-50 border-purple-200 mb-6">
      <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4 mb-6">
        <div>
          <h3 className="text-lg font-bold text-purple-800 mb-1">🤖 FAQ Agent Runs</h3>
          <p className="text-sm text-purple-700">
            Start automated FAQ generation runs, monitor live progress, and stop active runs cleanly.
          </p>
        </div>
        <button onClick={() => loadRuns()} className="btn-secondary text-sm self-start">
          ↻ Refresh Runs
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="bg-white border border-purple-100 rounded-lg p-5">
          <h4 className="font-semibold text-gray-800 mb-4">Start a New Run</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <label className="text-sm">
              <span className="block text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1">
                Collection
              </span>
              <select
                value={selectedCollection}
                onChange={(event) => setSelectedCollection(event.target.value)}
                className="w-full form-input px-3 py-2 rounded-md text-sm"
              >
                <option value="">— Select collection —</option>
                {documentCollections.map((collection) => {
                  const id = collection.alias || collection.name;
                  return (
                    <option key={id} value={id}>
                      {id} ({collection.count} pts)
                    </option>
                  );
                })}
              </select>
            </label>

            <NumberField
              label="Document limit"
              value={form.limit_documents}
              min={1}
              max={500}
              onChange={handleNumberField('limit_documents')}
            />
            <NumberField
              label="Max hops"
              value={form.max_hops}
              min={0}
              max={3}
              onChange={handleNumberField('max_hops')}
            />
            <NumberField
              label="Linked docs / source"
              value={form.max_linked_documents}
              min={0}
              max={10}
              onChange={handleNumberField('max_linked_documents')}
            />
            <NumberField
              label="FAQs / document"
              value={form.max_faqs_per_document}
              min={1}
              max={10}
              onChange={handleNumberField('max_faqs_per_document')}
            />

            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input type="checkbox" checked={form.follow_links} onChange={handleCheckboxField('follow_links')} />
              <span>Follow linked indexed documents</span>
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input type="checkbox" checked={form.force_reprocess} onChange={handleCheckboxField('force_reprocess')} />
              <span>Force reprocess unchanged docs</span>
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700 md:col-span-2">
              <input type="checkbox" checked={form.remove_stale_faqs} onChange={handleCheckboxField('remove_stale_faqs')} />
              <span>Remove stale FAQ sources not regenerated in this run</span>
            </label>
          </div>

          <div className="flex items-center gap-3 mt-5">
            <button
              onClick={handleStartRun}
              disabled={starting || !selectedCollection}
              className="btn-primary"
            >
              {starting ? 'Starting…' : 'Start FAQ Run'}
            </button>
            {activeRun && (
              <span className="text-sm text-gray-600">
                Active run: <span className="font-medium">{activeRun.run_id}</span>
              </span>
            )}
          </div>

          {message && <div className="mt-4 text-sm font-medium text-gray-700">{message}</div>}
        </div>

        <div className="bg-white border border-purple-100 rounded-lg p-5">
          <h4 className="font-semibold text-gray-800 mb-4">Run Overview</h4>
          {loading ? (
            <p className="text-sm text-gray-500">Loading runs…</p>
          ) : runs.length === 0 ? (
            <p className="text-sm text-gray-500">No FAQ runs yet.</p>
          ) : (
            <div className="space-y-3">
              <StatCell label="Total runs" value={runs.length} />
              <StatCell
                label="Active runs"
                value={runs.filter(isActiveRun).length}
              />
              <StatCell
                label="Completed runs"
                value={runs.filter((run) => run.status === 'completed').length}
              />
              <StatCell
                label="Cancelled runs"
                value={runs.filter((run) => run.status === 'cancelled').length}
              />
              <StatCell
                label="Failed runs"
                value={runs.filter((run) => run.status === 'failed').length}
              />
            </div>
          )}
        </div>
      </div>

      <div className="mt-6 space-y-4">
        {loading && runs.length === 0 ? (
          <p className="text-sm text-gray-500">Loading run details…</p>
        ) : runs.length === 0 ? (
          <p className="text-sm text-gray-500">Start a run to see progress and recent document activity.</p>
        ) : (
          runs.map(renderRunCard)
        )}
      </div>
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <label className="text-sm">
      <span className="block text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1">
        {label}
      </span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={onChange}
        className="w-full form-input px-3 py-2 rounded-md text-sm"
      />
    </label>
  );
}

function StatCell({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-gray-200 bg-gray-50 px-3 py-2">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">{label}</div>
      <div className="text-base font-semibold text-gray-800">{value}</div>
    </div>
  );
}
