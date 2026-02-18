/* ============================================================================
 * FAQ / KV Manager Tab
 *
 * Manage predefined FAQ entries per collection with semantic search,
 * CRUD operations, and relevance feedback.
 * ============================================================================ */

import { useCallback, useEffect, useState } from 'react';
import { apiFetch } from '../api/client';
import type { KVCollection, KVEntry } from '../types';
import { Modal, StarRating } from './ui';

export default function FaqTab() {
  const [collections, setCollections] = useState<KVCollection[]>([]);
  const [selected, setSelected] = useState('');
  const [entries, setEntries] = useState<KVEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [scoreThreshold, setScoreThreshold] = useState(0.5);
  const [isSearchResult, setIsSearchResult] = useState(false);
  const [resultCount, setResultCount] = useState('');
  const [newCollName, setNewCollName] = useState('');

  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [editId, setEditId] = useState('');
  const [editKey, setEditKey] = useState('');
  const [editValue, setEditValue] = useState('');
  const [modalStatus, setModalStatus] = useState('');

  const loadCollections = useCallback(async () => {
    try {
      const data = await apiFetch<KVCollection[]>('/kv');
      setCollections(data);
    } catch (e) {
      console.error('Failed to load KV collections:', e);
    }
  }, []);

  useEffect(() => {
    loadCollections();
  }, [loadCollections]);

  const loadEntries = useCallback(async () => {
    if (!selected) {
      setEntries([]);
      setResultCount('');
      return;
    }
    setLoading(true);
    setIsSearchResult(false);
    try {
      const data = await apiFetch<KVEntry[]>(`/kv/${encodeURIComponent(selected)}?limit=500`);
      setEntries(data);
      setResultCount(`${data.length} entries`);
    } catch (e) {
      console.error('Failed to load KV entries:', e);
    } finally {
      setLoading(false);
    }
  }, [selected]);

  useEffect(() => {
    loadEntries();
  }, [loadEntries]);

  const handleSearch = async () => {
    if (!searchQuery.trim() || !selected) {
      loadEntries();
      return;
    }
    setLoading(true);
    try {
      const data = await apiFetch<{ results: KVEntry[] }>(`/kv/${encodeURIComponent(selected)}/search`, {
        method: 'POST',
        body: JSON.stringify({ query: searchQuery.trim(), limit: 50, score_threshold: scoreThreshold }),
      });
      setEntries(data.results);
      setResultCount(`${data.results.length} results`);
      setIsSearchResult(true);
    } catch (e) {
      console.error('Search failed:', e);
    } finally {
      setLoading(false);
    }
  };

  const createCollection = async () => {
    if (!newCollName.trim()) return;
    try {
      const resp = await apiFetch<{ id?: string }>(`/kv/${encodeURIComponent(newCollName.trim())}`, {
        method: 'POST',
        body: JSON.stringify({ key: '__init__', value: '__init__' }),
      });
      if (resp.id) {
        await apiFetch(`/kv/${encodeURIComponent(newCollName.trim())}/${resp.id}`, { method: 'DELETE' });
      }
      setNewCollName('');
      await loadCollections();
      setSelected(newCollName.trim());
    } catch (e) {
      alert('Error creating collection: ' + (e instanceof Error ? e.message : e));
    }
  };

  const deleteEntry = async (id: string) => {
    if (!confirm('Delete this FAQ entry?')) return;
    try {
      await apiFetch(`/kv/${encodeURIComponent(selected)}/${id}`, { method: 'DELETE' });
      setEntries((prev) => prev.filter((e) => e.id !== id));
      loadCollections();
    } catch (e) {
      alert('Error: ' + (e instanceof Error ? e.message : e));
    }
  };

  const openCreate = () => {
    setEditId('');
    setEditKey('');
    setEditValue('');
    setModalStatus('');
    setModalOpen(true);
  };

  const openEdit = async (id: string) => {
    try {
      const entry = await apiFetch<KVEntry>(`/kv/${encodeURIComponent(selected)}/${id}`);
      setEditId(entry.id);
      setEditKey(entry.key);
      setEditValue(entry.value);
      setModalStatus('');
      setModalOpen(true);
    } catch (e) {
      alert('Error loading entry: ' + (e instanceof Error ? e.message : e));
    }
  };

  const saveEntry = async () => {
    if (!editKey.trim() || !editValue.trim()) {
      setModalStatus('Both key and value are required');
      return;
    }
    try {
      const body: Record<string, string> = { key: editKey.trim(), value: editValue.trim() };
      if (editId) body.id = editId;
      await apiFetch(`/kv/${encodeURIComponent(selected)}`, {
        method: 'POST',
        body: JSON.stringify(body),
      });
      setModalOpen(false);
      loadEntries();
      loadCollections();
    } catch (e) {
      setModalStatus('Error: ' + (e instanceof Error ? e.message : e));
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">📋 FAQ / Key-Value Manager</h2>
          <p className="text-sm text-gray-500 mt-1">
            Manage predefined FAQ entries per collection. Entries are matched via hybrid search.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            className="form-input px-3 py-2 rounded-md text-sm"
          >
            <option value="">— Select collection —</option>
            {collections.map((c) => (
              <option key={c.collection_name} value={c.collection_name}>
                {c.collection_name} ({c.count} entries)
              </option>
            ))}
          </select>
          <button onClick={loadCollections} className="btn-secondary text-sm" title="Refresh">
            ↻
          </button>
          <button onClick={openCreate} disabled={!selected} className="btn-primary text-sm">
            + New Entry
          </button>
        </div>
      </div>

      {/* New collection input */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-dashed border-gray-300">
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium text-gray-700 whitespace-nowrap">Create new collection:</label>
          <input
            type="text"
            value={newCollName}
            onChange={(e) => setNewCollName(e.target.value)}
            className="form-input flex-1 px-3 py-2 rounded-md text-sm"
            placeholder="e.g. my_chatbot_id"
          />
          <button onClick={createCollection} className="btn-primary text-sm">
            Create
          </button>
        </div>
      </div>

      {/* Search bar */}
      <div className="mb-6">
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            disabled={!selected}
            className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
            placeholder="Semantic search FAQ entries..."
          />
          <button onClick={handleSearch} disabled={!selected} className="absolute right-2 top-1/2 transform -translate-y-1/2 btn-primary px-4 py-1.5 rounded-md text-sm">
            Search
          </button>
        </div>
        <div className="flex items-center gap-4 mt-2 text-sm text-gray-500">
          <label className="flex items-center gap-2">
            <input
              type="number"
              value={scoreThreshold}
              onChange={(e) => setScoreThreshold(parseFloat(e.target.value) || 0.5)}
              min={0}
              max={1}
              step={0.05}
              className="w-20 px-2 py-1 border rounded text-sm"
            />
            <span>min score</span>
          </label>
          <span className="ml-auto">{resultCount}</span>
        </div>
      </div>

      {/* Entries list */}
      {loading ? (
        <div className="text-center py-8">
          <div className="loading-spinner mx-auto mb-2" />
          <p className="text-sm text-gray-500">Loading...</p>
        </div>
      ) : entries.length === 0 ? (
        <p className="text-gray-400 text-center py-12">
          {selected ? 'No entries found' : 'Select a collection to view FAQ entries'}
        </p>
      ) : (
        <div className="space-y-3">
          {entries.map((entry) => (
            <KVEntryCard
              key={entry.id}
              entry={entry}
              showScore={isSearchResult}
              searchQuery={searchQuery}
              collection={selected}
              onEdit={() => openEdit(entry.id)}
              onDelete={() => deleteEntry(entry.id)}
            />
          ))}
        </div>
      )}

      {/* Create/Edit modal */}
      <Modal open={modalOpen} onClose={() => setModalOpen(false)} title={editId ? 'Edit FAQ Entry' : 'New FAQ Entry'}>
        <div className="mb-4">
          <label className="block text-sm font-semibold text-gray-700 mb-1">Key / Question</label>
          <textarea
            value={editKey}
            onChange={(e) => setEditKey(e.target.value)}
            rows={3}
            className="w-full form-input px-4 py-3 rounded-md"
            placeholder="e.g. What are the opening hours?"
          />
        </div>
        <div className="mb-6">
          <label className="block text-sm font-semibold text-gray-700 mb-1">Value / Answer</label>
          <textarea
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            rows={6}
            className="w-full form-input px-4 py-3 rounded-md"
            placeholder="e.g. We are open Monday to Friday from 8:00 to 16:00."
          />
        </div>
        <div className="flex justify-end gap-3">
          <button onClick={() => setModalOpen(false)} className="btn-secondary">Cancel</button>
          <button onClick={saveEntry} className="btn-primary">Save</button>
        </div>
        {modalStatus && <div className="mt-3 text-sm text-red-600">{modalStatus}</div>}
      </Modal>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Single KV entry card                                                       */
/* -------------------------------------------------------------------------- */

function KVEntryCard({
  entry,
  showScore,
  searchQuery,
  collection,
  onEdit,
  onDelete,
}: {
  entry: KVEntry;
  showScore: boolean;
  searchQuery: string;
  collection: string;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const [disabled, setDisabled] = useState(false);

  const submitFeedback = async (rating: number, rank?: number) => {
    if (!searchQuery.trim()) return;
    setDisabled(true);
    setFeedbackStatus('Submitting...');
    try {
      const body: Record<string, unknown> = {
        query: searchQuery.trim(),
        kv_id: entry.id,
        kv_key: entry.key,
        kv_value: entry.value,
        search_score: entry.score || 0,
        user_rating: rating,
      };
      if (rank != null) body.ranking_score = rank;
      await apiFetch(`/kv/${encodeURIComponent(collection)}/feedback`, {
        method: 'POST',
        body: JSON.stringify(body),
      });
      setFeedbackStatus(rank != null ? `✓ Ranked ${rank}/5` : rating === 1 ? '✓ Relevant' : '✓ Irrelevant');
    } catch {
      setFeedbackStatus('Error');
      setDisabled(false);
    }
  };

  return (
    <div className="border rounded-lg p-4 hover:shadow-md transition-shadow bg-gray-50">
      <div className="flex justify-between items-start gap-4 mb-2">
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-gray-800 text-base">{entry.key}</div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {showScore && entry.score != null && (
            <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded font-mono">
              Score: {entry.score.toFixed(3)}
            </span>
          )}
          <button onClick={onEdit} className="btn-secondary text-xs px-3 py-1">Edit</button>
          <button onClick={onDelete} className="btn-danger text-xs px-3 py-1">Delete</button>
        </div>
      </div>
      <p className="text-sm text-gray-600 whitespace-pre-wrap">{entry.value}</p>
      <div className="flex gap-4 mt-2 text-xs text-gray-400">
        <span>ID: {entry.id}</span>
        {entry.updated_at && <span>Updated: {new Date(entry.updated_at).toLocaleString()}</span>}
      </div>

      {/* Feedback row (only for search results) */}
      {showScore && (
        <div className="mt-3 flex items-center gap-2 border-t pt-2 flex-wrap">
          <span className="text-xs text-gray-500">Relevant?</span>
          <button disabled={disabled} onClick={() => submitFeedback(1)} className="px-2 py-1 rounded text-xs bg-green-100 hover:bg-green-200 text-green-700 disabled:opacity-50">👍</button>
          <button disabled={disabled} onClick={() => submitFeedback(-1)} className="px-2 py-1 rounded text-xs bg-red-100 hover:bg-red-200 text-red-700 disabled:opacity-50">👎</button>
          <span className="text-xs text-gray-400 mx-1">|</span>
          <span className="text-xs text-gray-500">Rank:</span>
          <StarRating disabled={disabled} onRate={(s) => submitFeedback(s >= 4 ? 1 : s <= 2 ? -1 : 0, s)} />
          <span className="text-xs text-gray-400 ml-2">{feedbackStatus}</span>
        </div>
      )}
    </div>
  );
}
