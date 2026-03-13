/* ============================================================================
 * Recent Documents Panel — Shows recently indexed documents
 * ============================================================================ */

import { useCallback, useEffect, useState } from 'react';
import { apiFetch } from '../api/client';
import { useApp } from '../store';
import type { AdminDocumentItem } from '../types';

interface RecentDocumentsPanelProps {
  className?: string;
  listClassName?: string;
}

export default function RecentDocumentsPanel({
  className = '',
  listClassName = '',
}: RecentDocumentsPanelProps) {
  const { currentCollection } = useApp();
  const [documents, setDocuments] = useState<AdminDocumentItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadRecentDocuments = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const params = new URLSearchParams({
        limit: '10',
        recent_first: 'true',
      });
      if (currentCollection) {
        params.set('collection_name', currentCollection);
      }

      const response = await apiFetch<{ items: AdminDocumentItem[]; total: number }>(
        `/admin/documents?${params.toString()}`
      );
      setDocuments(response.items || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  }, [currentCollection]);

  useEffect(() => {
    void loadRecentDocuments();
  }, [loadRecentDocuments]);

  return (
    <div className={`bg-white rounded-lg shadow-sm border ${className}`}>
      <div className="bg-gray-100 px-4 py-3 border-b flex justify-between items-center">
        <h3 className="font-semibold text-gray-800">📄 Recent Documents</h3>
        <button
          onClick={() => void loadRecentDocuments()}
          className="text-xs px-2 py-1 bg-white border rounded hover:bg-gray-50"
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      <div className={`p-4 ${listClassName}`}>
        {error && (
          <div className="text-sm text-red-600 bg-red-50 p-3 rounded mb-3">
            Error loading documents: {error}
          </div>
        )}

        {loading && documents.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <div className="loading-spinner mx-auto mb-2" />
            <p className="text-sm">Loading recent documents...</p>
          </div>
        ) : documents.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <p className="text-sm">No documents found</p>
          </div>
        ) : (
          <div className="space-y-3">
            {documents.map((doc) => (
              <div
                key={doc.doc_id}
                className="border-l-2 border-blue-400 pl-3 py-2 hover:bg-gray-50 rounded-r transition-colors"
              >
                <a
                  href={doc.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-sm font-medium text-blue-600 hover:underline block mb-1 break-all"
                >
                  {doc.url || doc.doc_id}
                </a>
                <p className="text-xs text-gray-600 line-clamp-2">
                  {doc.content_preview || 'No preview available'}
                </p>
                <div className="flex gap-3 mt-1 text-xs text-gray-500">
                  {doc.indexed_at && (
                    <span>
                      {new Date(doc.indexed_at).toLocaleDateString(undefined, {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </span>
                  )}
                  {doc.faqs_count > 0 && (
                    <span className="text-blue-600">
                      {doc.faqs_count} FAQ{doc.faqs_count !== 1 ? 's' : ''}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
