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

// --- Tab types ---

export type TabName = 'search' | 'faq' | 'quality' | 'maintenance';
