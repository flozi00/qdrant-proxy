import { useCallback, useEffect, useState } from 'react';
import { marked } from 'marked';
import { apiFetch } from '../api/client';
import { useApp } from '../store';
import type { DocumentDetail, DocumentListResponse, DocumentSummary } from '../types';
import { Modal, Spinner } from './ui';

const RECENT_DOCUMENT_LIMIT = 10;
const AUTO_REFRESH_MS = 5000;

interface RecentDocumentsPanelProps {
  className?: string;
  listClassName?: string;
}

function getDomain(document: DocumentSummary): string {
  const metadataDomain = document.metadata?.domain;
  if (typeof metadataDomain === 'string' && metadataDomain.trim()) {
    return metadataDomain;
  }

  try {
    return new URL(document.url).hostname;
  } catch {
    return 'unknown-domain';
  }
}

function formatIndexedAt(indexedAt?: string): string {
  if (!indexedAt) return 'No indexed_at metadata';

  const date = new Date(indexedAt);
  if (Number.isNaN(date.getTime())) return indexedAt;
  return date.toLocaleString();
}

export default function RecentDocumentsPanel({ className = '', listClassName = '' }: RecentDocumentsPanelProps) {
  const { currentCollection } = useApp();
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedDocument, setSelectedDocument] = useState<DocumentDetail | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);

  const loadRecentDocuments = useCallback(async (silent = false) => {
    if (!currentCollection) {
      setDocuments([]);
      setTotal(0);
      setError('');
      return;
    }

    if (!silent) setLoading(true);
    setError('');

    try {
      const response = await apiFetch<DocumentListResponse>(
        `/admin/documents?collection_name=${encodeURIComponent(currentCollection)}&limit=${RECENT_DOCUMENT_LIMIT}&recent_first=true`,
      );
      setDocuments(response.items || []);
      setTotal(response.total || 0);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (!silent) setLoading(false);
    }
  }, [currentCollection]);

  const openDocument = useCallback(async (docId: string) => {
    if (!currentCollection) return;

    setDetailLoading(true);
    try {
      const detail = await apiFetch<DocumentDetail>(
        `/admin/documents/${encodeURIComponent(docId)}?collection_name=${encodeURIComponent(currentCollection)}`,
      );
      setSelectedDocument(detail);
      setDetailOpen(true);
    } catch (err) {
      alert('Failed to load document: ' + (err instanceof Error ? err.message : err));
    } finally {
      setDetailLoading(false);
    }
  }, [currentCollection]);

  useEffect(() => {
    void loadRecentDocuments();
  }, [loadRecentDocuments]);

  useEffect(() => {
    if (!autoRefresh || !currentCollection) return undefined;

    const intervalId = window.setInterval(() => {
      void loadRecentDocuments(true);
    }, AUTO_REFRESH_MS);

    return () => window.clearInterval(intervalId);
  }, [autoRefresh, currentCollection, loadRecentDocuments]);

  return (
    <div className={`bg-white rounded-lg shadow-md border border-emerald-100 overflow-hidden ${className}`.trim()}>
      <div className="px-6 py-5 bg-gradient-to-r from-emerald-50 via-white to-cyan-50 border-b border-emerald-100">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h3 className="text-xl font-bold text-gray-900">Recent Index Activity</h3>
            <p className="text-sm text-gray-600 mt-1">
              Watch the latest indexed documents land in <span className="font-semibold text-emerald-700">{currentCollection || 'no collection selected'}</span>.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <span className="px-3 py-1 rounded-full bg-emerald-100 text-emerald-800 font-semibold">
              Showing {documents.length} of {total}
            </span>
            <label className="flex items-center gap-2 text-gray-600 cursor-pointer">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="h-4 w-4 text-emerald-600"
              />
              <span>Auto refresh every 5s</span>
            </label>
            <button onClick={() => void loadRecentDocuments()} className="btn-secondary text-sm">
              Refresh now
            </button>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-gray-500">
          <span>{autoRefresh ? 'Live watch enabled' : 'Live watch paused'}</span>
          {lastUpdated && <span>Last updated: {lastUpdated}</span>}
          {error && <span className="text-red-600">{error}</span>}
        </div>
      </div>

      {!currentCollection ? (
        <div className="p-6 text-sm text-gray-500">Select a document collection to inspect recent indexing activity.</div>
      ) : loading && documents.length === 0 ? (
        <Spinner label="Loading recent documents..." />
      ) : documents.length === 0 ? (
        <div className="p-6 text-sm text-gray-500">No indexed documents found in this collection yet.</div>
      ) : (
        <div className={`divide-y divide-gray-100 ${listClassName}`.trim()}>
          {documents.map((document) => (
            <article key={document.doc_id} className="p-6 hover:bg-gray-50 transition-colors">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2 text-xs mb-2">
                    <span className="px-2.5 py-1 rounded-full bg-cyan-100 text-cyan-800 font-semibold">
                      {getDomain(document)}
                    </span>
                    <span className="px-2.5 py-1 rounded-full bg-amber-100 text-amber-800 font-semibold">
                      {document.faqs_count} FAQ{document.faqs_count === 1 ? '' : 's'}
                    </span>
                    <span className="text-gray-500">Indexed: {formatIndexedAt(document.indexed_at)}</span>
                  </div>
                  <div className="text-sm font-medium text-gray-900 break-all">{document.url}</div>
                  <p className="mt-2 text-sm text-gray-600 whitespace-pre-wrap break-words">
                    {document.content_preview || 'No content preview available.'}
                  </p>
                  <div className="mt-3 text-xs text-gray-400 font-mono break-all">
                    {document.doc_id}
                  </div>
                </div>
                <div className="flex items-center gap-3 shrink-0">
                  <a
                    href={document.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-sm text-blue-600 hover:text-blue-800 hover:underline"
                  >
                    Open source
                  </a>
                  <button
                    onClick={() => void openDocument(document.doc_id)}
                    className="btn-primary text-sm"
                    disabled={detailLoading}
                  >
                    {detailLoading && selectedDocument?.doc_id !== document.doc_id ? 'Loading...' : 'Inspect'}
                  </button>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}

      <Modal
        open={detailOpen}
        onClose={() => setDetailOpen(false)}
        title={selectedDocument ? 'Indexed Document Detail' : 'Loading document'}
        maxWidth="max-w-5xl"
      >
        {selectedDocument ? (
          <div>
            <div className="flex flex-col gap-2 text-sm text-gray-600 mb-4">
              <div className="break-all">
                <span className="font-semibold text-gray-800">URL:</span>{' '}
                <a href={selectedDocument.url} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
                  {selectedDocument.url}
                </a>
              </div>
              <div className="font-mono text-xs text-gray-500 break-all">{selectedDocument.doc_id}</div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)] gap-6">
              <div>
                <h4 className="font-semibold text-gray-900 mb-2">Content</h4>
                <div
                  className="bg-gray-50 border border-gray-200 rounded-lg p-4 max-h-[65vh] overflow-y-auto markdown-content"
                  dangerouslySetInnerHTML={{ __html: marked.parse(selectedDocument.content || '') as string }}
                />
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 mb-2">Metadata</h4>
                <pre className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-xs overflow-x-auto mb-4">
                  {JSON.stringify(selectedDocument.metadata || {}, null, 2)}
                </pre>
                <h4 className="font-semibold text-gray-900 mb-2">Extracted FAQ Entries ({selectedDocument.faqs_count})</h4>
                <div className="space-y-2 max-h-72 overflow-y-auto">
                  {selectedDocument.faqs.length === 0 ? (
                    <p className="text-sm text-gray-500">No FAQ entries extracted yet.</p>
                  ) : (
                    selectedDocument.faqs.map((faq) => (
                      <div key={faq.id} className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                        <div className="font-medium text-gray-800">Q: {faq.question}</div>
                        <div className="mt-1 text-sm text-gray-600">A: {faq.answer}</div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <Spinner label="Loading document..." />
        )}
      </Modal>
    </div>
  );
}