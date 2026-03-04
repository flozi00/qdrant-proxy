/* ============================================================================
 * Search Tab — Knowledge Base search, Web search, URL Fetch
 *
 * Four search/ingest modes with result display, markdown preview, and feedback.
 * ============================================================================ */

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { marked } from 'marked';
import { apiFetch } from '../api/client';
import { mcpClient } from '../api/mcp';
import { useApp } from '../store';
import type {
  DocumentDetail,
  FAQEntry,
  LLMDocumentRankingHint,
  LLMRankDocumentOption,
  LLMSearchRankingResponse,
  SearchDocument,
} from '../types';
import { urlToDocId } from '../utils';
import { FAQEntryDisplay, Modal, StarRating } from './ui';

type SearchMode = 'qdrant';

/* -------------------------------------------------------------------------- */
/* Types for FAQ generation                                                   */
/* -------------------------------------------------------------------------- */

interface DuplicateCandidate {
  id: string;
  question: string;
  answer: string;
  score: number;
  source_documents: { document_id: string; url?: string }[];
  source_count: number;
}

interface GenerateFAQResponse {
  question: string;
  answer: string;
  duplicates: DuplicateCandidate[];
}

interface QueryMatchSummary {
  phrase: string;
  phraseCount: number;
  termCounts: Record<string, number>;
  totalTermHits: number;
  bestChain: { text: string; count: number; length: number } | null;
}

interface QueuedQuery {
  id: string;
  query: string;
  source: string;
  created_at: string;
}

type MatchFilterMode = 'word' | 'chain' | 'word_or_chain';

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function parseQueryTerms(query: string): string[] {
  return query
    .split(/\s+/)
    .map((t) => t.trim())
    .filter(Boolean)
    .map((t) => t.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ''))
    .filter(Boolean);
}

function countOccurrences(haystack: string, needle: string): number {
  const source = haystack.trim();
  const target = needle.trim();
  if (!source || !target) return 0;
  const regex = new RegExp(escapeRegex(target), 'gi');
  return source.match(regex)?.length || 0;
}

function analyzeQueryMatches(query: string, text: string): QueryMatchSummary {
  const phrase = query.trim();
  const terms = parseQueryTerms(phrase);
  const uniqueTerms = Array.from(new Set(terms.map((t) => t.toLowerCase())));
  const phraseCount = countOccurrences(text, phrase);
  const termCounts = uniqueTerms.reduce<Record<string, number>>((acc, term) => {
    acc[term] = countOccurrences(text, term);
    return acc;
  }, {});

  let bestChain: QueryMatchSummary['bestChain'] = null;
  for (let chainLength = terms.length; chainLength >= 2; chainLength -= 1) {
    let chainFound = false;
    for (let i = 0; i <= terms.length - chainLength; i += 1) {
      const chain = terms.slice(i, i + chainLength).join(' ').trim();
      const chainCount = countOccurrences(text, chain);
      if (chainCount > 0) {
        bestChain = {
          text: chain,
          count: chainCount,
          length: chainLength,
        };
        chainFound = true;
        break;
      }
    }
    if (chainFound) break;
  }

  const totalTermHits = Object.values(termCounts).reduce((sum, count) => sum + count, 0);

  return {
    phrase,
    phraseCount,
    termCounts,
    totalTermHits,
    bestChain,
  };
}

function highlightByQuery(text: string, query: string): ReactNode[] {
  const phrase = query.trim();
  const terms = parseQueryTerms(phrase);
  const patterns = Array.from(
    new Set([
      ...(phrase ? [phrase] : []),
      ...terms,
    ]),
  )
    .filter(Boolean)
    .sort((a, b) => b.length - a.length);

  if (!patterns.length || !text) return [text];

  const regex = new RegExp(`(${patterns.map((p) => escapeRegex(p)).join('|')})`, 'gi');
  return text.split(regex).map((part, idx) => {
    if (idx % 2 === 1) {
      return (
        <mark key={`${part}-${idx}`} className="bg-yellow-200 text-gray-900 px-0.5 rounded-sm">
          {part}
        </mark>
      );
    }
    return part;
  });
}

function buildHighlightPatterns(query: string): string[] {
  const phrase = query.trim();
  const terms = parseQueryTerms(phrase);
  return Array.from(
    new Set([
      ...(phrase ? [phrase] : []),
      ...terms,
    ]),
  )
    .filter(Boolean)
    .sort((a, b) => b.length - a.length);
}

function highlightMarkdownHtmlByQuery(html: string, query: string): string {
  const patterns = buildHighlightPatterns(query);
  if (!html || patterns.length === 0 || typeof document === 'undefined') return html;

  const regex = new RegExp(`(${patterns.map((p) => escapeRegex(p)).join('|')})`, 'gi');
  const template = document.createElement('template');
  template.innerHTML = html;

  const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_TEXT);
  const textNodes: Text[] = [];
  let current = walker.nextNode();
  while (current) {
    textNodes.push(current as Text);
    current = walker.nextNode();
  }

  textNodes.forEach((textNode) => {
    const source = textNode.nodeValue || '';
    if (!source.trim() || !regex.test(source)) {
      regex.lastIndex = 0;
      return;
    }
    regex.lastIndex = 0;

    const parts = source.split(regex);
    if (parts.length <= 1) return;

    const fragment = document.createDocumentFragment();
    parts.forEach((part, idx) => {
      if (!part) return;
      if (idx % 2 === 1) {
        const mark = document.createElement('mark');
        mark.className = 'bg-yellow-200 text-gray-900 px-0.5 rounded-sm';
        mark.textContent = part;
        fragment.appendChild(mark);
      } else {
        fragment.appendChild(document.createTextNode(part));
      }
    });

    textNode.parentNode?.replaceChild(fragment, textNode);
  });

  return template.innerHTML;
}

function passesMatchFilter(
  summary: QueryMatchSummary,
  mode: MatchFilterMode,
  minCountInclusive: number,
): boolean {
  if (minCountInclusive <= 0) return true;
  const maxWordCount = Object.values(summary.termCounts).reduce(
    (max, count) => (count > max ? count : max),
    0,
  );
  const chainCount = Math.max(summary.phraseCount, summary.bestChain?.count || 0);

  if (mode === 'word') return maxWordCount >= minCountInclusive;
  if (mode === 'chain') return chainCount >= minCountInclusive;
  return maxWordCount >= minCountInclusive || chainCount >= minCountInclusive;
}

export default function SearchTab() {
  const [mode, setMode] = useState<SearchMode>('qdrant');

  return (
    <>
      {/* Hero card */}
      <div className="bg-white rounded-lg shadow-md p-8 mb-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-6">
            <h2 className="text-4xl font-bold text-blue-600 mb-2">🔍 Search &amp; Ingest</h2>
            <p className="text-gray-600">Search local knowledge base, web, or ingest a specific URL</p>
          </div>

          {/* Mode selector */}
          <div className="flex justify-center mb-6">
            <div className="inline-flex rounded-lg border border-gray-200 bg-gray-50 p-1">
              {([
                { id: 'qdrant', label: '📚 Knowledge Base', color: 'blue' },
              ] as const).map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMode(m.id)}
                  className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                    mode === m.id
                      ? `bg-${m.color}-600 text-white`
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {mode === 'qdrant' && <KnowledgeBaseSearch />}
        </div>
      </div>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Knowledge Base Search                                                      */
/* -------------------------------------------------------------------------- */

function KnowledgeBaseSearch() {
  const { refreshStats, currentCollection } = useApp();
  const [query, setQuery] = useState('');
  const [hybrid, setHybrid] = useState(true);
  const [limit, setLimit] = useState(10);
  const [faqs, setFaqs] = useState<FAQEntry[]>([]);
  const [docs, setDocs] = useState<SearchDocument[]>([]);
  const [previewContent, setPreviewContent] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [previewDocId, setPreviewDocId] = useState('');
  const [hasResults, setHasResults] = useState(false);
  const [docModalOpen, setDocModalOpen] = useState(false);
  const [docDetail, setDocDetail] = useState<DocumentDetail | null>(null);
  const [minMatchCount, setMinMatchCount] = useState(0);
  const [matchFilterMode, setMatchFilterMode] = useState<MatchFilterMode>('word_or_chain');
  const [queuedQueries, setQueuedQueries] = useState<QueuedQuery[]>([]);
  const [selectedQueueId, setSelectedQueueId] = useState('');
  const [queueLoading, setQueueLoading] = useState(false);
  const [llmHintsById, setLlmHintsById] = useState<Record<string, LLMDocumentRankingHint>>({});
  const [llmHintModel, setLlmHintModel] = useState('');
  const [llmHintsLoading, setLlmHintsLoading] = useState(false);
  const [llmHintsError, setLlmHintsError] = useState('');
  const lastQuery = useRef('');

  // FAQ generation state for preview panel
  const [previewSelectedText, setPreviewSelectedText] = useState('');
  const [previewFaqModal, setPreviewFaqModal] = useState(false);
  const [previewGenerating, setPreviewGenerating] = useState(false);
  const [previewFaqResult, setPreviewFaqResult] = useState<GenerateFAQResponse | null>(null);
  const [previewEditQ, setPreviewEditQ] = useState('');
  const [previewEditA, setPreviewEditA] = useState('');
  const [previewSubmitting, setPreviewSubmitting] = useState(false);
  const [previewSubmitStatus, setPreviewSubmitStatus] = useState('');

  const search = useCallback(async (overrideQuery?: string) => {
    const effectiveQuery = (overrideQuery ?? query).trim();
    if (!effectiveQuery) return;
    lastQuery.current = effectiveQuery;
    try {
      const result = await mcpClient.callTool<{ faqs?: FAQEntry[]; documents?: SearchDocument[] }>(
        'search_knowledge_base',
        { query: effectiveQuery, limit, collection_name: currentCollection || undefined },
      );
      const resultDocs = result?.documents || [];
      setQuery(effectiveQuery);
      setFaqs(result?.faqs || []);
      setDocs(resultDocs);
      setLlmHintsById({});
      setLlmHintModel('');
      setLlmHintsError('');
      setHasResults(true);
      setPreviewContent('');

      if (resultDocs.length > 0) {
        setLlmHintsLoading(true);
        const options: LLMRankDocumentOption[] = resultDocs.map((doc, idx) => ({
          option_id: doc.doc_id || doc.url || `doc-${idx + 1}`,
          doc_id: doc.doc_id,
          url: doc.url,
          content: doc.content,
          search_score: doc.score,
        }));

        try {
          const ranking = await apiFetch<LLMSearchRankingResponse>('/admin/search/llm-rank', {
            method: 'POST',
            body: JSON.stringify({
              query: effectiveQuery,
              documents: options,
            }),
          });
          const nextHints: Record<string, LLMDocumentRankingHint> = {};
          for (const hint of ranking.hints || []) {
            nextHints[hint.option_id] = hint;
          }
          setLlmHintsById(nextHints);
          setLlmHintModel(ranking.model || '');
        } catch (err) {
          setLlmHintsError(err instanceof Error ? err.message : String(err));
        } finally {
          setLlmHintsLoading(false);
        }
      }
    } catch (err) {
      alert('Search failed: ' + (err instanceof Error ? err.message : err));
      setLlmHintsLoading(false);
    }
  }, [query, limit, hybrid, currentCollection]);

  const loadQueuedQueries = useCallback(async () => {
    setQueueLoading(true);
    try {
      const res = await apiFetch<{ items?: QueuedQuery[] }>(
        `/admin/query-queue?collection_name=${encodeURIComponent(currentCollection || '')}&limit=100`,
      );
      setQueuedQueries(res.items || []);
    } catch {
      setQueuedQueries([]);
    } finally {
      setQueueLoading(false);
    }
  }, [currentCollection]);

  useEffect(() => {
    void loadQueuedQueries();
  }, [loadQueuedQueries]);

  const replaySelectedQuery = useCallback(async () => {
    if (!selectedQueueId) return;
    const selected = queuedQueries.find((item) => item.id === selectedQueueId);
    if (!selected) return;

    try {
      await search(selected.query);
      await apiFetch(`/admin/query-queue/${encodeURIComponent(selected.id)}`, {
        method: 'DELETE',
      });
      setSelectedQueueId('');
      await loadQueuedQueries();
    } catch (err) {
      alert('Replay failed: ' + (err instanceof Error ? err.message : err));
    }
  }, [selectedQueueId, queuedQueries, search, loadQueuedQueries]);

  const openPreview = (doc: SearchDocument) => {
    setPreviewContent(doc.content);
    setPreviewUrl(doc.url);
    setPreviewDocId(doc.doc_id);
    setPreviewSelectedText('');
    setPreviewFaqResult(null);
    setPreviewSubmitStatus('');
  };

  const viewDocDetail = async (docId: string, collection: string) => {
    try {
      const detail = await apiFetch<DocumentDetail>(
        `/admin/documents/${docId}?collection_name=${encodeURIComponent(collection)}`,
      );
      setDocDetail(detail);
      setDocModalOpen(true);
    } catch (err) {
      alert('Failed to load document: ' + (err instanceof Error ? err.message : err));
    }
  };

  const extractFAQs = async (docId: string, collection: string) => {
    if (!confirm('This will delete existing FAQ entries and re-extract new ones. Continue?')) return;
    try {
      const result = await apiFetch<{ deleted_facts?: { deleted?: number }; facts_extracted: number }>(
        `/admin/documents/${docId}/extract?collection_name=${encodeURIComponent(collection)}`,
        { method: 'POST' },
      );
      const deleted = typeof result.deleted_facts === 'object' ? result.deleted_facts?.deleted || 0 : 0;
      alert(`Extraction complete!\nDeleted: ${deleted} old FAQ entries\nExtracted: ${result.facts_extracted} new FAQ entries`);
      refreshStats();
    } catch (err) {
      alert('Error: ' + (err instanceof Error ? err.message : err));
    }
  };

  const filteredFaqs = useMemo(() => {
    if (!lastQuery.current.trim() || minMatchCount <= 0) return faqs;
    return faqs.filter((faq) => {
      const summary = analyzeQueryMatches(lastQuery.current, `${faq.question}\n${faq.answer}`);
      return passesMatchFilter(summary, matchFilterMode, minMatchCount);
    });
  }, [faqs, minMatchCount, matchFilterMode]);

  const filteredDocs = useMemo(() => {
    if (!lastQuery.current.trim() || minMatchCount <= 0) return docs;
    return docs.filter((doc) => {
      const summary = analyzeQueryMatches(lastQuery.current, doc.content || '');
      return passesMatchFilter(summary, matchFilterMode, minMatchCount);
    });
  }, [docs, minMatchCount, matchFilterMode]);

  return (
    <>
      {/* Search input */}
      <div className="relative mb-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && search()}
          className="w-full px-6 py-4 text-lg border-2 border-gray-300 rounded-full focus:border-blue-500 focus:outline-none shadow-sm hover:shadow-md transition-shadow"
          placeholder="Search documents and FAQ entries..."
        />
        <button onClick={() => void search()} className="absolute right-2 top-1/2 transform -translate-y-1/2 btn-primary px-6 py-2 rounded-full">
          Search
        </button>
      </div>

      <div className="flex items-center gap-4 text-sm text-gray-600">
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={hybrid} onChange={(e) => setHybrid(e.target.checked)} className="h-4 w-4 text-blue-600" />
          <span>Hybrid search (MCP default)</span>
        </label>
        <span className="text-gray-400">|</span>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="number"
            value={limit}
            onChange={(e) => setLimit(parseInt(e.target.value) || 10)}
            min={1}
            max={50}
            className="w-16 px-2 py-1 border rounded"
          />
          <span>results</span>
        </label>
      </div>

      <div className="mt-3 p-3 bg-gray-50 border border-gray-200 rounded-lg">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-gray-700">Queued Replay Queries</span>
          <button
            onClick={() => void loadQueuedQueries()}
            className="text-xs px-2 py-1 border rounded bg-white hover:bg-gray-100"
          >
            Refresh
          </button>
          {queueLoading && <span className="text-xs text-gray-500">Loading...</span>}
        </div>
        <div className="mt-2 flex flex-wrap gap-2 items-center">
          <select
            value={selectedQueueId}
            onChange={(e) => setSelectedQueueId(e.target.value)}
            className="min-w-[320px] max-w-full px-2 py-1 border rounded bg-white text-sm"
          >
            <option value="">Select queued query...</option>
            {queuedQueries.map((item) => (
              <option key={item.id} value={item.id}>
                {item.query} ({item.source})
              </option>
            ))}
          </select>
          <button
            onClick={() => void replaySelectedQuery()}
            disabled={!selectedQueueId}
            className="btn-primary text-xs disabled:opacity-50"
          >
            Replay And Remove
          </button>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-gray-600">
        <span className="font-medium text-gray-700">Match filter</span>
        <span className="text-gray-500">Show only when count &gt;= X</span>
        <label className="flex items-center gap-2">
          <span className="text-gray-600">X</span>
          <input
            type="number"
            min={0}
            value={minMatchCount}
            onChange={(e) => setMinMatchCount(Math.max(0, parseInt(e.target.value) || 0))}
            className="w-20 px-2 py-1 border rounded"
          />
        </label>
        <select
          value={matchFilterMode}
          onChange={(e) => setMatchFilterMode(e.target.value as MatchFilterMode)}
          className="px-2 py-1 border rounded bg-white"
        >
          <option value="word_or_chain">Word OR chain</option>
          <option value="word">Word only</option>
          <option value="chain">Chain only</option>
        </select>
        {hasResults && minMatchCount > 0 && (
          <span className="text-xs text-gray-500">
            FAQ: {filteredFaqs.length}/{faqs.length}, Docs: {filteredDocs.length}/{docs.length}
          </span>
        )}
      </div>

      {hasResults && (
        <div className="mt-2 text-xs text-gray-500">
          {llmHintsLoading && <span>LLM rating hints are being generated...</span>}
          {!llmHintsLoading && llmHintModel && (
            <span>LLM hint model: <code>{llmHintModel}</code></span>
          )}
          {!llmHintsLoading && llmHintsError && (
            <span className="text-amber-700">LLM hints unavailable: {llmHintsError}</span>
          )}
        </div>
      )}

      {/* Results */}
      {hasResults && (
        <div className="flex flex-col md:flex-row gap-6 items-start mt-6">
          {/* Left: results list */}
          <div className="w-full md:w-1/2 space-y-6">
            {/* FAQ Entries */}
            {filteredFaqs.length > 0 && (
              <div className="bg-blue-50 border-l-4 border-blue-500 rounded-lg shadow-sm p-6">
                <h3 className="text-lg font-semibold text-blue-900 mb-4">FAQ Entries</h3>
                <div className="space-y-3">
                  {filteredFaqs.map((faq, i) => (
                    <FAQResult key={`${lastQuery.current}:${faq.id || i}`} faq={faq} searchQuery={lastQuery.current} />
                  ))}
                </div>
              </div>
            )}

            {/* Documents */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Documents</h3>
              {filteredDocs.length === 0 ? (
                <p className="text-gray-600 text-center py-8">No documents found.</p>
              ) : (
                <div className="space-y-4">
                  {filteredDocs.map((doc, i) => (
                    <DocResult
                      key={`${lastQuery.current}:${doc.doc_id || i}`}
                      doc={doc}
                      onPreview={openPreview}
                      searchQuery={lastQuery.current}
                      llmHint={llmHintsById[doc.doc_id || doc.url || `doc-${i + 1}`]}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right: preview */}
          <div className="w-full md:w-1/2 sticky top-8">
            <div className="bg-white rounded-lg shadow-lg border-2 border-blue-100 overflow-hidden flex flex-col" style={{ maxHeight: 'calc(100vh - 100px)' }}>
              <div className="bg-blue-600 px-4 py-3 flex justify-between items-center text-white">
                <h3 className="font-bold">Markdown Preview</h3>
                <div className="flex items-center gap-2">
                  {previewSelectedText && previewUrl && (
                    <button
                      onClick={() => {
                        setPreviewFaqModal(true);
                        setPreviewFaqResult(null);
                        setPreviewSubmitStatus('');
                        // Auto-generate on open
                        (async () => {
                          setPreviewGenerating(true);
                          try {
                            const result = await apiFetch<GenerateFAQResponse>('/admin/documents/generate-faq', {
                              method: 'POST',
                              body: JSON.stringify({
                                selected_text: previewSelectedText,
                                source_url: previewUrl,
                                document_id: previewDocId || undefined,
                                collection_name: currentCollection,
                              }),
                            });
                            setPreviewFaqResult(result);
                            setPreviewEditQ(result.question);
                            setPreviewEditA(result.answer);
                          } catch (err) {
                            alert('FAQ generation failed: ' + (err instanceof Error ? err.message : err));
                            setPreviewFaqModal(false);
                          } finally {
                            setPreviewGenerating(false);
                          }
                        })();
                      }}
                      className="px-3 py-1 bg-amber-500 hover:bg-amber-600 text-white text-xs rounded font-medium"
                    >
                      🤖 Generate FAQ
                    </button>
                  )}
                  <button onClick={() => { setPreviewContent(''); setPreviewSelectedText(''); }} className="hover:bg-blue-700 rounded p-1">✕</button>
                </div>
              </div>
              {previewSelectedText && (
                <div className="px-4 py-2 bg-amber-50 border-b border-amber-200 text-xs text-amber-800">
                  Selected: "{previewSelectedText.substring(0, 100)}{previewSelectedText.length > 100 ? '…' : ''}"
                </div>
              )}
              <div
                className="p-6 overflow-y-auto markdown-content text-sm text-gray-800 bg-gray-50 flex-1 cursor-text select-text"
                onMouseUp={() => {
                  const sel = window.getSelection();
                  const text = sel?.toString().trim() || '';
                  if (text.length >= 10) setPreviewSelectedText(text);
                }}
              >
                {previewContent ? (
                  <div
                    dangerouslySetInnerHTML={{
                      __html: highlightMarkdownHtmlByQuery(marked.parse(previewContent) as string, lastQuery.current),
                    }}
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-gray-400 py-20">
                    <p>Select a result to preview its content here.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Document detail modal */}
      <Modal open={docModalOpen} onClose={() => setDocModalOpen(false)} title="Document Details" maxWidth="max-w-4xl">
        {docDetail && <DocumentDetailView doc={docDetail} onExtractFAQs={extractFAQs} />}
      </Modal>

      {/* Preview FAQ generation modal */}
      <Modal
        open={previewFaqModal}
        onClose={() => setPreviewFaqModal(false)}
        title="Generate FAQ from Selection"
        maxWidth="max-w-2xl"
      >
        {previewGenerating ? (
          <div className="text-center py-8">
            <div className="loading-spinner mx-auto mb-2" />
            <p className="text-sm text-gray-500">Generating FAQ with LLM...</p>
          </div>
        ) : previewFaqResult ? (
          <div className="space-y-4">
            <div className="p-3 bg-gray-50 rounded text-xs text-gray-600">
              <strong>Source:</strong>{' '}
              <a href={previewUrl} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
                {previewUrl}
              </a>
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1">Question</label>
              <textarea
                value={previewEditQ}
                onChange={(e) => setPreviewEditQ(e.target.value)}
                rows={2}
                className="w-full form-input px-3 py-2 rounded-md"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1">Answer</label>
              <textarea
                value={previewEditA}
                onChange={(e) => setPreviewEditA(e.target.value)}
                rows={4}
                className="w-full form-input px-3 py-2 rounded-md"
              />
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={async () => {
                  if (!previewEditQ.trim() || !previewEditA.trim()) return;
                  setPreviewSubmitting(true);
                  setPreviewSubmitStatus('');
                  try {
                    const result = await apiFetch<{ ok: boolean; action: string; faq_id: string }>('/admin/documents/submit-faq', {
                      method: 'POST',
                      body: JSON.stringify({
                        question: previewEditQ.trim(),
                        answer: previewEditA.trim(),
                        source_url: previewUrl,
                        document_id: previewDocId || undefined,
                        collection_name: currentCollection,
                      }),
                    });
                    setPreviewSubmitStatus(`✓ Created new FAQ entry`);
                    setTimeout(() => { setPreviewFaqModal(false); setPreviewSelectedText(''); }, 1500);
                  } catch (err) {
                    setPreviewSubmitStatus('✗ Error: ' + (err instanceof Error ? err.message : err));
                  } finally {
                    setPreviewSubmitting(false);
                  }
                }}
                disabled={previewSubmitting || !previewEditQ.trim() || !previewEditA.trim()}
                className="btn-success text-sm"
              >
                {previewSubmitting ? 'Submitting...' : '✓ Create New FAQ'}
              </button>
              <button onClick={() => setPreviewFaqModal(false)} className="btn-secondary text-sm">
                Cancel
              </button>
            </div>
            {previewSubmitStatus && (
              <div className={`text-sm ${previewSubmitStatus.startsWith('✓') ? 'text-green-600' : 'text-red-600'}`}>
                {previewSubmitStatus}
              </div>
            )}

            {/* Duplicate candidates */}
            {previewFaqResult.duplicates.length > 0 && (
              <div className="border-t pt-3">
                <h5 className="text-sm font-semibold text-gray-700 mb-2">
                  🔍 Similar existing FAQs — merge source instead?
                </h5>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {previewFaqResult.duplicates.map((dup) => (
                    <div key={dup.id} className="bg-gray-50 p-3 rounded border hover:shadow-sm transition-shadow">
                      <div className="flex justify-between items-start gap-2">
                        <div className="flex-1 min-w-0">
                          <FAQEntryDisplay question={dup.question} answer={dup.answer} />
                          <div className="mt-1 flex gap-3 text-xs text-gray-400">
                            <span>Score: {dup.score.toFixed(3)}</span>
                            <span>{dup.source_count} source{dup.source_count !== 1 ? 's' : ''}</span>
                          </div>
                        </div>
                        <button
                          onClick={async () => {
                            setPreviewSubmitting(true);
                            setPreviewSubmitStatus('');
                            try {
                              const result = await apiFetch<{ ok: boolean; action: string; source_count?: number }>('/admin/documents/submit-faq', {
                                method: 'POST',
                                body: JSON.stringify({
                                  question: previewEditQ.trim(),
                                  answer: previewEditA.trim(),
                                  source_url: previewUrl,
                                  document_id: previewDocId || undefined,
                                  collection_name: currentCollection,
                                  merge_with_id: dup.id,
                                }),
                              });
                              setPreviewSubmitStatus(`✓ Merged — source added (${result.source_count} total)`);
                              setTimeout(() => { setPreviewFaqModal(false); setPreviewSelectedText(''); }, 1500);
                            } catch (err) {
                              setPreviewSubmitStatus('✗ Error: ' + (err instanceof Error ? err.message : err));
                            } finally {
                              setPreviewSubmitting(false);
                            }
                          }}
                          disabled={previewSubmitting}
                          className="btn-primary text-xs px-3 py-1 flex-shrink-0"
                        >
                          Merge Source
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : null}
      </Modal>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* FAQ search result with feedback                                            */
/* -------------------------------------------------------------------------- */

function FAQResult({ faq, searchQuery }: { faq: FAQEntry; searchQuery: string }) {
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const [disabled, setDisabled] = useState(false);
  const searchableText = `${faq.question}\n${faq.answer}`;
  const matchSummary = useMemo(
    () => analyzeQueryMatches(searchQuery, searchableText),
    [searchQuery, searchableText],
  );

  useEffect(() => {
    setFeedbackStatus('');
    setDisabled(false);
  }, [searchQuery]);

  const submitFeedback = async (rating: number, rank?: number) => {
    if (!searchQuery) return;
    setDisabled(true);
    setFeedbackStatus('Submitting...');
    try {
      const body: Record<string, unknown> = {
        query: searchQuery,
        faq_id: faq.id,
        faq_text: `Q: ${faq.question}\nA: ${faq.answer}`.substring(0, 500),
        search_score: faq.score || 0,
        user_rating: rating,
        content_type: 'faq',
      };
      if (rank != null) body.ranking_score = rank;
      await apiFetch('/feedback', { method: 'POST', body: JSON.stringify(body) });
      setFeedbackStatus(rank != null ? `✓ Ranked ${rank}/5` : rating === 1 ? '✓ Relevant' : '✓ Irrelevant');
    } catch {
      setFeedbackStatus('Error');
      setDisabled(false);
    }
  };

  return (
    <div className="bg-white p-4 rounded border-l-4 border-blue-400">
      <div className="flex flex-col gap-1">
        <div className="font-medium text-gray-700 break-words">Q: {highlightByQuery(faq.question, searchQuery)}</div>
        <div className="text-gray-600 break-words">A: {highlightByQuery(faq.answer, searchQuery)}</div>
      </div>
      {faq.score != null && <span className="ml-auto text-xs text-gray-500 float-right">Score: {faq.score.toFixed(3)}</span>}
      {(matchSummary.totalTermHits > 0 || matchSummary.phraseCount > 0) && (
        <div className="mt-2 text-xs text-gray-700 flex flex-wrap gap-2">
          <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-900">
            Phrase: {matchSummary.phraseCount}
          </span>
          {Object.entries(matchSummary.termCounts).map(([term, count]) => (
            <span key={term} className="px-2 py-0.5 rounded bg-sky-100 text-sky-900">
              {term}: {count}
            </span>
          ))}
          {matchSummary.bestChain && (
            <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-900">
              Best chain ({matchSummary.bestChain.length} words): {matchSummary.bestChain.count}
            </span>
          )}
        </div>
      )}
      {faq.source_documents && faq.source_documents.length > 0 && (
        <div className="mt-2 text-xs text-gray-500">
          Sources:{' '}
          {faq.source_documents.map((s, i) => {
            try {
              const hostname = s.url ? new URL(s.url).hostname : 'unknown';
              return (
                <span key={i}>
                  {i > 0 && ', '}
                  <a href={s.url || '#'} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
                    {hostname}
                  </a>
                </span>
              );
            } catch {
              return <span key={i}>{i > 0 && ', '}{s.document_id || 'unknown'}</span>;
            }
          })}
        </div>
      )}

      {/* Feedback row */}
      <div className="mt-3 flex items-center gap-2 border-t pt-2 flex-wrap">
        <span className="text-xs text-gray-500">Relevant?</span>
        <button disabled={disabled} onClick={() => submitFeedback(1)} className="px-2 py-1 rounded text-xs bg-green-100 hover:bg-green-200 text-green-700 disabled:opacity-50">👍</button>
        <button disabled={disabled} onClick={() => submitFeedback(-1)} className="px-2 py-1 rounded text-xs bg-red-100 hover:bg-red-200 text-red-700 disabled:opacity-50">👎</button>
        <span className="text-xs text-gray-400 mx-1">|</span>
        <span className="text-xs text-gray-500">Rank:</span>
        <StarRating
          disabled={disabled}
          onRate={(s) => submitFeedback(s >= 4 ? 1 : s <= 2 ? -1 : 0, s)}
        />
        <span className="text-xs text-gray-400 ml-2">{feedbackStatus}</span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Document search result with feedback                                       */
/* -------------------------------------------------------------------------- */

function DocResult({
  doc,
  onPreview,
  searchQuery,
  llmHint,
}: {
  doc: SearchDocument;
  onPreview: (doc: SearchDocument) => void;
  searchQuery: string;
  llmHint?: LLMDocumentRankingHint;
}) {
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const [disabled, setDisabled] = useState(false);
  const matchSummary = useMemo(
    () => analyzeQueryMatches(searchQuery, doc.content || ''),
    [searchQuery, doc.content],
  );
  const previewText = useMemo(() => {
    if (!doc.content) return '';
    if (doc.content.length <= 300) return doc.content;
    return `${doc.content.substring(0, 300)}...`;
  }, [doc.content]);

  useEffect(() => {
    setFeedbackStatus('');
    setDisabled(false);
  }, [searchQuery]);

  const submitFeedback = async (rating: number, rank?: number) => {
    if (!searchQuery) return;
    setDisabled(true);
    setFeedbackStatus('Submitting...');
    try {
      let docId = doc.doc_id;
      if (!docId && doc.url) docId = urlToDocId(doc.url);
      const body: Record<string, unknown> = {
        query: searchQuery,
        doc_id: docId,
        doc_url: doc.url,
        doc_content: doc.content,
        search_score: doc.score,
        user_rating: rating,
        content_type: 'document',
      };
      if (rank != null) body.ranking_score = rank;
      await apiFetch('/feedback', { method: 'POST', body: JSON.stringify(body) });
      setFeedbackStatus(rank != null ? `✓ Ranked ${rank}/5` : rating === 1 ? '✓ Relevant' : '✓ Irrelevant');
    } catch {
      setFeedbackStatus('Error');
      setDisabled(false);
    }
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-2 gap-4">
        <a href={doc.url} target="_blank" rel="noreferrer" className="text-lg font-semibold text-blue-600 hover:underline block break-all flex-1 min-w-0">
          {doc.url}
        </a>
        <span className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded flex-shrink-0">
          Score: {doc.score.toFixed(3)}
        </span>
      </div>
      {(matchSummary.totalTermHits > 0 || matchSummary.phraseCount > 0) && (
        <div className="mb-2 text-xs text-gray-700 flex flex-wrap gap-2">
          <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-900">
            Phrase: {matchSummary.phraseCount}
          </span>
          {Object.entries(matchSummary.termCounts).map(([term, count]) => (
            <span key={term} className="px-2 py-0.5 rounded bg-sky-100 text-sky-900">
              {term}: {count}
            </span>
          ))}
          {matchSummary.bestChain && (
            <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-900">
              Best chain ({matchSummary.bestChain.length} words): {matchSummary.bestChain.count}
            </span>
          )}
        </div>
      )}
      <p className="text-sm text-gray-700 mb-2 break-words overflow-hidden">
        {highlightByQuery(previewText, searchQuery)}
      </p>
      <div className="flex gap-3 items-center">
        <button onClick={() => onPreview(doc)} className="btn-primary text-xs">Preview</button>
        <a href={doc.url} target="_blank" rel="noreferrer" className="text-sm text-blue-600 hover:underline">Open Link</a>
      </div>

      {llmHint && (
        <div className="mt-3 p-2 rounded border border-indigo-200 bg-indigo-50 text-xs text-indigo-900">
          <div className="font-semibold">
            LLM rating hint: {'★'.repeat(Math.max(1, Math.min(5, llmHint.stars)))}{'☆'.repeat(5 - Math.max(1, Math.min(5, llmHint.stars)))}
            {' '}({llmHint.stars}/5), rank #{llmHint.relative_rank}
          </div>
          <div className="mt-1 text-indigo-800">{llmHint.reason}</div>
          <div className="mt-1 text-indigo-700">Use this as orientation for your manual star feedback below.</div>
        </div>
      )}

      {/* Feedback row */}
      <div className="mt-3 flex items-center gap-2 border-t pt-2 flex-wrap">
        <span className="text-xs text-gray-500">Relevant?</span>
        <button disabled={disabled} onClick={() => submitFeedback(1)} className="px-2 py-1 rounded text-xs bg-green-100 hover:bg-green-200 text-green-700 disabled:opacity-50">👍</button>
        <button disabled={disabled} onClick={() => submitFeedback(-1)} className="px-2 py-1 rounded text-xs bg-red-100 hover:bg-red-200 text-red-700 disabled:opacity-50">👎</button>
        <span className="text-xs text-gray-400 mx-1">|</span>
        <span className="text-xs text-gray-500">Rank:</span>
        <StarRating disabled={disabled} onRate={(s) => submitFeedback(s >= 4 ? 1 : s <= 2 ? -1 : 0, s)} />
        <span className="text-xs text-gray-400 ml-2">{feedbackStatus}</span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Document detail view (inside modal)                                        */
/* -------------------------------------------------------------------------- */

function DocumentDetailView({
  doc,
  onExtractFAQs,
}: {
  doc: DocumentDetail;
  onExtractFAQs: (docId: string, collection: string) => void;
}) {
  const { currentCollection } = useApp();
  const contentRef = useRef<HTMLDivElement>(null);
  const [selectedText, setSelectedText] = useState('');
  const [generating, setGenerating] = useState(false);
  const [faqResult, setFaqResult] = useState<GenerateFAQResponse | null>(null);
  const [editQuestion, setEditQuestion] = useState('');
  const [editAnswer, setEditAnswer] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState('');

  /** Capture text selection from the content area */
  const handleMouseUp = () => {
    const sel = window.getSelection();
    const text = sel?.toString().trim() || '';
    if (text.length >= 10) {
      setSelectedText(text);
    }
  };

  /** Call LLM to generate a FAQ from the selected text */
  const generateFAQ = async () => {
    if (!selectedText || selectedText.length < 10) return;
    setGenerating(true);
    setFaqResult(null);
    setSubmitStatus('');
    try {
      const result = await apiFetch<GenerateFAQResponse>('/admin/documents/generate-faq', {
        method: 'POST',
        body: JSON.stringify({
          selected_text: selectedText,
          source_url: doc.url,
          document_id: doc.doc_id,
          collection_name: currentCollection,
        }),
      });
      setFaqResult(result);
      setEditQuestion(result.question);
      setEditAnswer(result.answer);
    } catch (err) {
      alert('FAQ generation failed: ' + (err instanceof Error ? err.message : err));
    } finally {
      setGenerating(false);
    }
  };

  /** Submit a new FAQ entry */
  const submitFAQ = async (mergeWithId?: string) => {
    if (!editQuestion.trim() || !editAnswer.trim()) return;
    setSubmitting(true);
    setSubmitStatus('');
    try {
      const result = await apiFetch<{ ok: boolean; action: string; faq_id: string; source_count?: number }>(
        '/admin/documents/submit-faq',
        {
          method: 'POST',
          body: JSON.stringify({
            question: editQuestion.trim(),
            answer: editAnswer.trim(),
            source_url: doc.url,
            document_id: doc.doc_id,
            collection_name: currentCollection,
            merge_with_id: mergeWithId || null,
          }),
        },
      );
      if (result.action === 'merged') {
        setSubmitStatus(`✓ Merged — source added (${result.source_count} total sources)`);
      } else {
        setSubmitStatus(`✓ Created new FAQ entry`);
      }
      // Reset for next selection
      setSelectedText('');
      setFaqResult(null);
    } catch (err) {
      setSubmitStatus('✗ Error: ' + (err instanceof Error ? err.message : err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <p className="text-sm text-gray-600 mb-1">
        Document ID: <code className="bg-gray-100 px-2 py-1 rounded">{doc.doc_id}</code>
      </p>
      <p className="text-sm text-gray-600 mb-4">
        URL:{' '}
        <a href={doc.url} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
          {doc.url}
        </a>
      </p>

      {/* FAQ generation hint */}
      <div className="mb-3 p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">
        <strong>💡 Generate FAQ:</strong> Select text in the content below, then click "Generate FAQ from Selection" to create a Q&A pair.
      </div>

      <h4 className="font-semibold mb-2">Content</h4>
      <div
        ref={contentRef}
        onMouseUp={handleMouseUp}
        className="bg-gray-50 p-4 rounded max-h-96 overflow-y-auto markdown-content mb-2 cursor-text select-text"
        dangerouslySetInnerHTML={{ __html: marked.parse(doc.content) as string }}
      />

      {/* Selection & Generate bar */}
      <div className="flex items-center gap-3 mb-4">
        {selectedText ? (
          <>
            <span className="text-xs text-gray-500 truncate max-w-xs" title={selectedText}>
              Selected: "{selectedText.substring(0, 80)}{selectedText.length > 80 ? '…' : ''}"
            </span>
            <button
              onClick={generateFAQ}
              disabled={generating}
              className="btn-primary text-sm flex items-center gap-2"
            >
              {generating ? (
                <>
                  <span className="loading-spinner" style={{ width: 14, height: 14 }} />
                  Generating...
                </>
              ) : (
                '🤖 Generate FAQ from Selection'
              )}
            </button>
            <button onClick={() => { setSelectedText(''); setFaqResult(null); setSubmitStatus(''); }} className="btn-secondary text-xs">
              Clear
            </button>
          </>
        ) : (
          <span className="text-xs text-gray-400 italic">Select text above to generate a FAQ entry</span>
        )}
      </div>

      {/* Generated FAQ review panel */}
      {faqResult && (
        <div className="mb-6 border-2 border-blue-200 rounded-lg overflow-hidden">
          <div className="bg-blue-600 px-4 py-2 text-white font-semibold text-sm">
            Generated FAQ — Review & Edit
          </div>
          <div className="p-4 space-y-3">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1">Question</label>
              <textarea
                value={editQuestion}
                onChange={(e) => setEditQuestion(e.target.value)}
                rows={2}
                className="w-full form-input px-3 py-2 rounded-md text-sm"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1">Answer</label>
              <textarea
                value={editAnswer}
                onChange={(e) => setEditAnswer(e.target.value)}
                rows={4}
                className="w-full form-input px-3 py-2 rounded-md text-sm"
              />
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => submitFAQ()}
                disabled={submitting || !editQuestion.trim() || !editAnswer.trim()}
                className="btn-success text-sm"
              >
                {submitting ? 'Submitting...' : '✓ Create New FAQ'}
              </button>
              <button
                onClick={() => { setFaqResult(null); setSubmitStatus(''); }}
                className="btn-secondary text-sm"
              >
                Discard
              </button>
            </div>
            {submitStatus && (
              <div className={`text-sm ${submitStatus.startsWith('✓') ? 'text-green-600' : 'text-red-600'}`}>
                {submitStatus}
              </div>
            )}

            {/* Duplicate candidates */}
            {faqResult.duplicates.length > 0 && (
              <div className="mt-4 border-t pt-3">
                <h5 className="text-sm font-semibold text-gray-700 mb-2">
                  🔍 Similar existing FAQs — merge source instead?
                </h5>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {faqResult.duplicates.map((dup) => (
                    <div key={dup.id} className="bg-gray-50 p-3 rounded border hover:shadow-sm transition-shadow">
                      <div className="flex justify-between items-start gap-2">
                        <div className="flex-1 min-w-0">
                          <FAQEntryDisplay question={dup.question} answer={dup.answer} />
                          <div className="mt-1 flex gap-3 text-xs text-gray-400">
                            <span>Score: {dup.score.toFixed(3)}</span>
                            <span>{dup.source_count} source{dup.source_count !== 1 ? 's' : ''}</span>
                          </div>
                        </div>
                        <button
                          onClick={() => submitFAQ(dup.id)}
                          disabled={submitting}
                          className="btn-primary text-xs px-3 py-1 flex-shrink-0"
                          title="Add this document as a source to the existing FAQ"
                        >
                          Merge Source
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <h4 className="font-semibold mb-2">Metadata</h4>
      <pre className="bg-gray-50 p-4 rounded text-sm overflow-x-auto mb-4">{JSON.stringify(doc.metadata, null, 2)}</pre>

      <div className="flex justify-between items-center mb-2">
        <h4 className="font-semibold">Extracted FAQ Entries ({doc.faqs_count})</h4>
        <button onClick={() => onExtractFAQs(doc.doc_id, currentCollection)} className="btn-success text-sm">
          {doc.faqs_count > 0 ? 'Re-extract FAQs' : 'Extract FAQs'}
        </button>
      </div>
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {doc.faqs.length === 0 ? (
          <p className="text-gray-500 text-sm italic">No FAQ entries extracted yet.</p>
        ) : (
          doc.faqs.map((f) => (
            <div key={f.id} className="bg-gray-50 p-3 rounded">
              <FAQEntryDisplay question={f.question} answer={f.answer} />
            </div>
          ))
        )}
      </div>
    </>
  );
}

