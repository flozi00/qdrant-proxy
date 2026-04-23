/* ============================================================================
 * TypeScript types for the Qdrant Proxy Admin UI
 * ============================================================================ */

// --- Collections ---

export interface CollectionInfo {
  name: string;
  count: number;
  type: 'documents' | 'faq';
  alias?: string;
}

export interface AdminStats {
  collections: CollectionInfo[];
  total_documents: number;
  total_faqs: number;
}

// --- FAQ Entries ---

export interface FAQEntry {
  id: string;
  question: string;
  answer: string;
  score?: number;
  source_documents?: { document_id: string; url?: string }[];
  source_count?: number;
  aggregated_confidence?: number;
}

// --- Documents ---

export interface DocumentSummary {
  doc_id: string;
  url: string;
  content_preview: string;
  faqs_count: number;
  indexed_at?: string;
  metadata: Record<string, unknown>;
}

export interface DocumentDetail {
  doc_id: string;
  url: string;
  content: string;
  metadata: Record<string, unknown>;
  faqs_count: number;
  faqs: FAQEntry[];
}

export interface DocumentListResponse {
  items: DocumentSummary[];
  total: number;
  next_offset?: string | null;
}

// --- Search ---

export interface SearchResult {
  faqs: FAQEntry[];
  documents: SearchDocument[];
}

export interface SearchDocument {
  doc_id: string;
  url: string;
  content: string;
  score: number;
}

export interface LLMRankDocumentOption {
  option_id: string;
  doc_id?: string;
  url: string;
  content: string;
  search_score: number;
}

export interface LLMDocumentRankingHint {
  option_id: string;
  stars: number;
  relative_rank: number;
  reason: string;
}

export interface LLMSearchRankingResponse {
  query: string;
  model: string;
  hints: LLMDocumentRankingHint[];
}

export interface WebSearchResult {
  url: string;
  title: string;
  description: string;
}

// --- FAQ / KV ---

export interface KVCollection {
  collection_name: string;
  count: number;
}

export interface KVEntry {
  id: string;
  key: string;
  value: string;
  score?: number;
  updated_at?: string;
}

// --- Feedback ---

export interface FeedbackResponse {
  id: string;
  query: string;
  faq_id?: string;
  faq_text?: string;
  doc_id?: string;
  doc_url?: string;
  doc_content?: string;
  search_score: number;
  user_rating: number;
  ranking_score?: number;
  rating_session_id?: string;
  content_type: string;
  collection_name: string;
  created_at: string;
}

export interface FeedbackStats {
  collection_name: string;
  total_feedback: number;
  positive_feedback?: number;
  neutral_feedback?: number;
  negative_feedback?: number;
  score_threshold_recommendations?: Recommendation[];
  common_failure_patterns?: FailurePattern[];
}

export interface Recommendation {
  type: string;
  message: string;
  rationale?: string;
  requires_human_approval?: boolean;
}

export interface FailurePattern {
  pattern: string;
  occurrences: number;
  suggestion?: string;
}

export interface FeedbackExport {
  format: string;
  total_records: number;
  positive_pairs: number;
  negative_pairs: number;
  contrastive_pairs: number;
  binary_pairs?: number;
  ranked_pairs?: number;
  data: Record<string, unknown>[];
}

// --- Maintenance ---

export interface ModelConfig {
  dense_model_id: string;
  colbert_model_id: string;
  dense_vector_size: number;
}

export interface MaintenanceTask {
  status: string;
  total: number;
  completed: number;
  batch_size: number;
  vector_types: string[];
  source_collection: string;
  target_collection: string;
  alias_name: string;
  start_time: string;
  end_time?: string;
  error?: string;
}

export interface FAQAgentRecentDocument {
  doc_id: string;
  url?: string;
  status: string;
  generated_faq_count?: number;
  error?: string;
}

export interface FAQAgentRunRequest {
  collection_name?: string;
  limit_documents: number;
  follow_links: boolean;
  max_hops: number;
  max_linked_documents: number;
  max_faqs_per_document: number;
  force_reprocess: boolean;
  remove_stale_faqs: boolean;
}

export interface FAQAgentRunResponse {
  run_id: string;
  collection_name: string;
  status: string;
  message: string;
}

export interface FAQAgentRunStatus {
  run_id: string;
  collection_name: string;
  status: string;
  limit_documents: number;
  follow_links: boolean;
  max_hops: number;
  max_linked_documents: number;
  max_faqs_per_document: number;
  force_reprocess: boolean;
  remove_stale_faqs: boolean;
  cancel_requested: boolean;
  documents_completed: number;
  documents_processed: number;
  documents_skipped: number;
  documents_failed: number;
  faqs_created: number;
  faqs_merged: number;
  faqs_refreshed: number;
  faqs_reassigned: number;
  faqs_removed_sources: number;
  faqs_deleted: number;
  current_document_id?: string | null;
  current_document_url?: string | null;
  handled_document_ids: string[];
  recent_documents: FAQAgentRecentDocument[];
  start_time: string;
  end_time?: string | null;
  error?: string | null;
}

export interface FAQAgentRunsResponse {
  items: FAQAgentRunStatus[];
}

// --- Tab types ---

export type TabName = 'search' | 'faq' | 'quality' | 'maintenance';
