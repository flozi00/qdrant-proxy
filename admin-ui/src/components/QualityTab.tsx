/* ============================================================================
 * Quality Feedback Tab
 *
 * Sub-tabs:
 *   - Search Feedback: stats, recommendations, feedback list, training export
 *   - FAQ Quality: stats, patterns, feedback list
 * ============================================================================ */

import { useCallback, useEffect, useState } from 'react';
import { marked } from 'marked';
import { apiFetch } from '../api/client';
import { useApp } from '../store';
import type { FeedbackExport, FeedbackResponse, FeedbackStats } from '../types';
import { Modal } from './ui';

type SubTab = 'search' | 'faq';

export default function QualityTab() {
  const [subTab, setSubTab] = useState<SubTab>('search');

  return (
    <>
      {/* Sub-tab navigation */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setSubTab('search')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${subTab === 'search' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-100 border'}`}
        >
          📊 Search Feedback
        </button>
        <button
          onClick={() => setSubTab('faq')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${subTab === 'faq' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-100 border'}`}
        >
          💡 FAQ Quality
        </button>
      </div>

      {subTab === 'search' && <SearchFeedbackSubTab />}
      {subTab === 'faq' && <FAQQualitySubTab />}
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Search Feedback Sub-Tab                                                    */
/* -------------------------------------------------------------------------- */

function SearchFeedbackSubTab() {
  const { currentCollection } = useApp();
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [feedbackList, setFeedbackList] = useState<FeedbackResponse[]>([]);
  const [filterRating, setFilterRating] = useState('');
  const [filterType, setFilterType] = useState('');
  const [contentModal, setContentModal] = useState<{ open: boolean; title: string; content: string; isDoc: boolean }>({
    open: false,
    title: '',
    content: '',
    isDoc: false,
  });

  const loadStats = useCallback(async () => {
    try {
      const url = currentCollection
        ? `/admin/feedback/stats?collection_name=${encodeURIComponent(currentCollection)}`
        : '/admin/feedback/stats';
      const data = await apiFetch<FeedbackStats>(url);
      setStats(data);
    } catch (e) {
      console.error('Failed to load feedback stats:', e);
    }
  }, [currentCollection]);

  const loadList = useCallback(async () => {
    try {
      let url = '/admin/feedback?limit=50';
      if (currentCollection) url += `&collection_name=${encodeURIComponent(currentCollection)}`;
      if (filterRating) url += `&user_rating=${filterRating}`;
      let items = await apiFetch<FeedbackResponse[]>(url);
      if (filterType) items = items.filter((f) => (f.content_type || 'faq') === filterType);
      setFeedbackList(items);
    } catch (e) {
      console.error('Failed to load feedback list:', e);
    }
  }, [currentCollection, filterRating, filterType]);

  useEffect(() => {
    loadStats();
    loadList();
  }, [loadStats, loadList]);

  const deleteFeedback = async (id: string) => {
    if (!confirm('Delete this feedback record?')) return;
    try {
      let url = `/admin/feedback/${id}`;
      if (currentCollection) url += `?collection_name=${encodeURIComponent(currentCollection)}`;
      await apiFetch(url, { method: 'DELETE' });
      loadList();
      loadStats();
    } catch (e) {
      alert('Error: ' + (e instanceof Error ? e.message : e));
    }
  };

  return (
    <>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Stats card */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold text-gray-800">📊 Feedback Statistics</h2>
            <button onClick={loadStats} className="btn-secondary text-sm">Refresh</button>
          </div>
          <div className="space-y-4">
            <div className="p-4 bg-gray-50 rounded text-center">
              <div className="text-3xl font-bold text-gray-800">{stats?.total_feedback ?? '-'}</div>
              <div className="text-sm text-gray-600">Total Feedback</div>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <StatBox value={stats?.positive_feedback} label="👍 Relevant" color="green" />
              <StatBox value={stats?.neutral_feedback} label="➖ Neutral" color="gray" />
              <StatBox value={stats?.negative_feedback} label="👎 Irrelevant" color="red" />
            </div>
          </div>
        </div>

        {/* Recommendations card */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold text-gray-800">💡 Recommendations</h2>
            <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">Human Review Required</span>
          </div>

          <div className="space-y-3">
            {stats?.score_threshold_recommendations?.length ? (
              stats.score_threshold_recommendations.map((rec, i) => (
                <div key={i} className="p-3 bg-yellow-50 border border-yellow-200 rounded">
                  <div className="font-medium text-yellow-800">{rec.type || 'Recommendation'}</div>
                  <p className="text-sm text-yellow-700">{rec.message}</p>
                  {rec.rationale && <p className="text-xs text-yellow-600 mt-1">{rec.rationale}</p>}
                  <p className="text-xs text-red-600 mt-1 font-medium">⚠️ Manual approval required</p>
                </div>
              ))
            ) : (
              <div className="p-4 bg-gray-50 rounded text-center text-gray-500">No recommendations yet</div>
            )}
          </div>

          <div className="mt-6">
            <h3 className="font-semibold text-gray-700 mb-3">🔍 Failure Patterns</h3>
            <div className="space-y-2 text-sm">
              {stats?.common_failure_patterns?.length ? (
                stats.common_failure_patterns.map((p, i) => (
                  <div key={i} className="p-2 bg-gray-50 rounded">
                    <span className="font-medium">{p.pattern}</span>
                    <span className="text-gray-500 ml-2">({p.occurrences} occurrences)</span>
                    {p.suggestion && <p className="text-xs text-gray-600 mt-1">{p.suggestion}</p>}
                  </div>
                ))
              ) : (
                <p className="text-gray-500">No patterns detected yet</p>
              )}
            </div>
          </div>

          <div className="mt-6 pt-4 border-t bg-yellow-50 -m-6 p-6 rounded-b-lg">
            <p className="text-sm text-yellow-800">
              <strong>⚠️ Important:</strong> All threshold adjustments and system changes must be made manually after human review.
            </p>
          </div>
        </div>
      </div>

      {/* Recent feedback list */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-gray-800">📝 Recent Feedback</h2>
          <div className="flex gap-2">
            <select value={filterRating} onChange={(e) => setFilterRating(e.target.value)} className="form-input px-3 py-2 rounded text-sm">
              <option value="">All Ratings</option>
              <option value="1">👍 Relevant</option>
              <option value="0">➖ Neutral</option>
              <option value="-1">👎 Irrelevant</option>
            </select>
            <select value={filterType} onChange={(e) => setFilterType(e.target.value)} className="form-input px-3 py-2 rounded text-sm">
              <option value="">All Types</option>
              <option value="faq">💡 FAQ Entries</option>
              <option value="document">📄 Documents</option>
            </select>
            <button onClick={loadList} className="btn-secondary text-sm">Refresh</button>
          </div>
        </div>

        {feedbackList.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No feedback found matching filters</p>
        ) : (
          <div className="space-y-3">
            {feedbackList.map((f) => (
              <FeedbackCard
                key={f.id}
                item={f}
                onDelete={() => deleteFeedback(f.id)}
                onShowFull={() => {
                  const isDoc = (f.content_type || 'faq') === 'document';
                  const content = isDoc ? f.doc_content || '' : f.faq_text || '';
                  setContentModal({ open: true, title: isDoc ? 'Full Document Content' : 'Full FAQ Text', content, isDoc });
                }}
              />
            ))}
          </div>
        )}
      </div>

      {/* Export section */}
      <ExportSection />

      {/* Full content modal */}
      <Modal open={contentModal.open} onClose={() => setContentModal((s) => ({ ...s, open: false }))} title={contentModal.title} maxWidth="max-w-4xl">
        {contentModal.isDoc ? (
          <div className="markdown-content" dangerouslySetInnerHTML={{ __html: marked.parse(contentModal.content) as string }} />
        ) : (
          <pre className="whitespace-pre-wrap text-sm text-gray-700">{contentModal.content}</pre>
        )}
      </Modal>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* FAQ Quality Sub-Tab                                                        */
/* -------------------------------------------------------------------------- */

function FAQQualitySubTab() {
  const { currentCollection } = useApp();
  const [stats, setStats] = useState<Record<string, number>>({});
  const [feedbackList, setFeedbackList] = useState<FeedbackResponse[]>([]);
  const [filterRating, setFilterRating] = useState('');

  const loadStats = useCallback(async () => {
    try {
      const url = currentCollection
        ? `/admin/feedback/stats?collection_name=${encodeURIComponent(currentCollection)}`
        : '/admin/feedback/stats';
      const data = await apiFetch<FeedbackStats>(url);
      setStats({
        total: data.total_feedback || 0,
        good: data.positive_feedback || 0,
        verbose: 0,
        incorrect: data.negative_feedback || 0,
        irrelevant: 0,
        duplicate: 0,
      });
    } catch (e) {
      console.error('Failed to load FAQ quality stats:', e);
    }
  }, [currentCollection]);

  const loadList = useCallback(async () => {
    try {
      let url = '/admin/feedback?limit=50';
      if (currentCollection) url += `&collection_name=${encodeURIComponent(currentCollection)}`;
      if (filterRating === 'good') url += '&user_rating=1';
      else if (filterRating === 'incorrect') url += '&user_rating=-1';
      let items = await apiFetch<FeedbackResponse[]>(url);
      items = items.filter((f) => (f.content_type || 'faq') === 'faq');
      setFeedbackList(items);
    } catch (e) {
      console.error('Failed to load FAQ quality list:', e);
    }
  }, [currentCollection, filterRating]);

  useEffect(() => {
    loadStats();
    loadList();
  }, [loadStats, loadList]);

  return (
    <>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Stats */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold text-gray-800">📊 FAQ Quality Stats</h2>
            <button onClick={loadStats} className="btn-secondary text-sm">Refresh</button>
          </div>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <StatBox value={stats.total} label="Total Feedback" color="gray" />
              <StatBox value={stats.good} label="✓ Good" color="green" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <StatBox value={stats.verbose} label="📝 Verbose" color="yellow" />
              <StatBox value={stats.incorrect} label="❌ Incorrect" color="red" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <StatBox value={stats.irrelevant} label="🚫 Irrelevant" color="gray" />
              <StatBox value={stats.duplicate} label="🔄 Duplicate" color="purple" />
            </div>
          </div>
        </div>

        {/* Top Subjects */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">🎯 Top Questions with Issues</h2>
          <p className="text-gray-500 text-sm">No data yet</p>
        </div>

        {/* Common Patterns */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">📋 Common Patterns</h2>
          <p className="text-gray-500 text-sm">No patterns detected yet</p>
        </div>
      </div>

      {/* FAQ quality feedback list */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-gray-800">📝 Recent FAQ Quality Feedback</h2>
          <div className="flex gap-2">
            <select value={filterRating} onChange={(e) => setFilterRating(e.target.value)} className="form-input px-3 py-2 rounded text-sm">
              <option value="">All Ratings</option>
              <option value="good">✓ Good</option>
              <option value="verbose">📝 Verbose</option>
              <option value="incorrect">❌ Incorrect</option>
              <option value="irrelevant">🚫 Irrelevant</option>
              <option value="duplicate">🔄 Duplicate</option>
            </select>
            <button onClick={loadList} className="btn-secondary text-sm">Refresh</button>
          </div>
        </div>

        {feedbackList.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No FAQ quality feedback recorded yet.</p>
        ) : (
          <div className="space-y-3">
            {feedbackList.map((f) => (
              <FeedbackCard key={f.id} item={f} onDelete={async () => {}} onShowFull={() => {}} />
            ))}
          </div>
        )}
      </div>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Shared components                                                          */
/* -------------------------------------------------------------------------- */

function StatBox({ value, label, color }: { value?: number; label: string; color: string }) {
  const bgMap: Record<string, string> = { green: 'bg-green-50', red: 'bg-red-50', gray: 'bg-gray-50', yellow: 'bg-yellow-50', purple: 'bg-purple-50' };
  const textMap: Record<string, string> = { green: 'text-green-600', red: 'text-red-600', gray: 'text-gray-600', yellow: 'text-yellow-600', purple: 'text-purple-600' };
  return (
    <div className={`p-3 ${bgMap[color] || 'bg-gray-50'} rounded text-center`}>
      <div className={`text-xl font-bold ${textMap[color] || 'text-gray-600'}`}>{value ?? '-'}</div>
      <div className="text-xs text-gray-600">{label}</div>
    </div>
  );
}

function FeedbackCard({
  item,
  onDelete,
  onShowFull,
}: {
  item: FeedbackResponse;
  onDelete: () => void;
  onShowFull: () => void;
}) {
  const isDoc = (item.content_type || 'faq') === 'document';
  const content = isDoc ? item.doc_content || '' : item.faq_text || '';
  const maxLen = isDoc ? 300 : 200;
  const truncated = content.substring(0, maxLen) + (content.length > maxLen ? '...' : '');
  const ratingEmoji = item.user_rating === 1 ? '👍' : item.user_rating === -1 ? '👎' : '➖';
  const borderColor = item.user_rating === 1 ? 'border-green-400' : item.user_rating === -1 ? 'border-red-400' : 'border-gray-400';
  const rankBadge =
    item.ranking_score != null
      ? `${'★'.repeat(item.ranking_score)}${'☆'.repeat(5 - item.ranking_score)}`
      : '';

  return (
    <div className={`p-4 bg-gray-50 rounded border-l-4 ${borderColor}`}>
      <div className="flex justify-between items-start mb-2">
        <div className="flex items-center gap-2">
          <span className="text-2xl">{ratingEmoji}</span>
          <span className="text-xl">{isDoc ? '📄' : '💡'}</span>
          <div>
            <div className="font-medium text-gray-800">{item.query}</div>
            <div className="text-xs text-gray-500 flex items-center gap-2">
              <span className={`px-2 py-0.5 ${isDoc ? 'bg-blue-100 text-blue-700' : 'bg-purple-100 text-purple-700'} text-xs rounded`}>
                {isDoc ? 'Document' : 'FAQ'}
              </span>
              {rankBadge && (
                <span className="px-2 py-0.5 bg-yellow-100 text-yellow-800 text-xs rounded font-medium">{rankBadge}</span>
              )}
              <span>Score: {item.search_score?.toFixed(3) || '-'}</span>
            </div>
          </div>
        </div>
        <div className="text-right text-xs text-gray-500">
          {item.created_at ? new Date(item.created_at).toLocaleString() : '-'}
        </div>
      </div>

      {isDoc && item.doc_url && (
        <div className="text-xs text-blue-600 mb-1">
          <a href={item.doc_url} target="_blank" rel="noreferrer">{item.doc_url}</a>
        </div>
      )}
      <p className="text-sm text-gray-600 mb-2">{truncated}</p>
      <div className="mt-2 flex justify-end gap-3">
        {content.length > maxLen && (
          <button onClick={onShowFull} className="text-xs text-blue-600 hover:underline">Show Full</button>
        )}
        <button onClick={onDelete} className="text-xs text-red-600 hover:underline">Delete</button>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Export Section                                                             */
/* -------------------------------------------------------------------------- */

function ExportSection() {
  const { currentCollection } = useApp();
  const [format, setFormat] = useState('contrastive');
  const [exportData, setExportData] = useState<FeedbackExport | null>(null);

  const doExport = async () => {
    try {
      let url = `/admin/feedback/export?format=${format}`;
      if (currentCollection) url += `&collection_name=${encodeURIComponent(currentCollection)}`;
      const data = await apiFetch<FeedbackExport>(url);
      setExportData(data);
    } catch (e) {
      alert('Export error: ' + (e instanceof Error ? e.message : e));
    }
  };

  const download = () => {
    if (!exportData?.data?.length) return;
    const blob = new Blob([exportData.data.map((d) => JSON.stringify(d)).join('\n')], { type: 'application/jsonl' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `feedback_export_${new Date().toISOString().split('T')[0]}.jsonl`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mt-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4">📤 Export Training Data</h2>
      <p className="text-gray-600 mb-4">Export user feedback as training data for embedding model fine-tuning.</p>
      <div className="flex flex-wrap gap-4 items-end">
        <div>
          <label className="text-sm text-gray-600">Format</label>
          <select value={format} onChange={(e) => setFormat(e.target.value)} className="form-input px-3 py-2 rounded text-sm block mt-1">
            <option value="contrastive">Contrastive (query + pos + neg)</option>
            <option value="jsonl">JSONL (raw)</option>
          </select>
        </div>
        <button onClick={doExport} className="btn-primary">Download Export</button>
      </div>

      {exportData && (
        <div className="mt-4 p-4 bg-gray-50 rounded">
          <div className="font-medium mb-2">Export Summary:</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>Total records: <strong>{exportData.total_records}</strong></div>
            <div>Positive pairs: <strong>{exportData.positive_pairs}</strong></div>
            <div>Negative pairs: <strong>{exportData.negative_pairs}</strong></div>
            <div>Contrastive triplets: <strong>{exportData.contrastive_pairs}</strong></div>
            {exportData.binary_pairs != null && <div>Binary pairs: <strong>{exportData.binary_pairs}</strong></div>}
            {exportData.ranked_pairs != null && <div>Ranked pairs: <strong>{exportData.ranked_pairs}</strong></div>}
          </div>
          {exportData.data?.length > 0 && (
            <div className="mt-3">
              <button onClick={download} className="btn-primary text-sm">Download JSON</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
