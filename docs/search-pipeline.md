# Search Pipeline

## Dual-Vector Embedding Strategy

```mermaid
flowchart LR
    subgraph Input
        Text["Document/Query Text"]
    end
    
    subgraph Encoding["Parallel Encoding"]
        ColBERT["ColBERT\n(ModernColBERT via vLLM)"]
        Dense["Dense\n(Qwen3-Embedding via vLLM)"]
    end
    
    subgraph Vectors["Vector Types"]
        MV["Multivector\n128-dim per token\nMaxSim scoring"]
        DV["Dense Vector\n1024-dim\nCosine similarity"]
    end
    
    Text --> ColBERT --> MV
    Text --> Dense --> DV
```

### Collection Schema

```python
vectors_config = {
    "colbert": VectorParams(
        size=128, 
        multivector_config=MaxSim,
        distance=Cosine
    ),
    "dense": VectorParams(
        size=1024, 
        distance=Cosine
    )
}
```

## Hybrid Search Flow

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Search
    participant Dense as Dense HNSW
    participant ColBERT as ColBERT MaxSim
    participant FAQs as FAQ Knowledge Base

    Client->>Search: POST /search (query, filters)
    
    rect rgb(200, 220, 255)
        Note over Search,Dense: Stage 1: Fast Prefetch
        Search->>Dense: query_dense (HNSW ef=128)
        Dense-->>Search: top-N candidates
    end
    
    rect rgb(220, 255, 220)
        Note over Search,ColBERT: Stage 2: Precise Rerank
        Search->>ColBERT: ColBERT MaxSim on candidates
        ColBERT-->>Search: Reranked results
    end
    
    rect rgb(255, 220, 220)
        Note over Search,FAQs: Stage 3: FAQ Enhancement
        Search->>FAQs: Search related FAQ entries
        FAQs-->>Search: Top 5 FAQ entries (score > 29.5)
        Search->>Search: Boost FAQ source URLs
    end
    
    Search->>Search: Apply time-based boosting
    Search->>Search: Transform scores (power=5)
    Search-->>Client: SearchResponse
```

### Time-Based Boosting Formula

```
score = original_score + exp_decay(current_time - doc_time)
```

Parameters:
- `scale_days`: Default 1 day
- `midpoint`: Default 0.5
- `datetime_field`: `metadata.indexed_at`

## Document Ingestion Flow

```mermaid
flowchart TD
    subgraph Input["Input"]
        URL["URL"]
        Content["Content"]
        File["File Upload"]
    end
    
    subgraph Acquisition["Content Acquisition"]
        Scrape["Scrape via native Docling\n(DocumentConverter)"]
        Parse["Parse File via native Docling\n(DocumentStream)"]
    end
    
    subgraph Embedding["Embedding Generation"]
        EncCol["encode_document()\nColBERT multivector"]
        EncDense["encode_dense()\nvLLM API → auto-dim"]
    end
    
    subgraph Storage["Storage"]
        Upsert["Upsert to Qdrant"]
    end
    
    URL --> Scrape --> Content
    File --> Parse --> Content
    Content --> EncCol & EncDense
    EncCol & EncDense --> Upsert
```

## Docling Integration

Docling is integrated natively as a Python library using `DocumentConverter`. The converter runs in-process with GPU acceleration and produces a `DoclingDocument` which is exported to a dict via `export_to_dict()`.

### File Type Handling

- **Plain-text formats** (`.txt`, `.csv`, `.json`, `.md`, `.yaml`, `.py`, `.js`, etc.) — detected by extension and read directly without Docling processing; structured formats are wrapped in code fences.
- **Binary/rich formats** (PDF, DOCX, PPTX, etc.) — processed through the full Docling pipeline.

### PDF Pipeline Options

- OCR enabled
- Accurate table structure detection
- `images_scale=2.0` for high-resolution picture rendering
- API-based picture description via `PictureDescriptionApiOptions` pointing to LiteLLM (`concurrency=4`)
- If conversion fails with picture enrichment, retries with simpler converter (no picture enrichment)

### Custom Markdown Renderer

- Keeps only `content_layer=body` nodes
- Skips low-signal captions
- Filters navigation-like lists
- Picture/chart nodes are checked before the generic text branch so VLM-generated descriptions are always emitted
- Descriptions wrapped in `[Image: ...]` brackets
- Descriptions read from `meta.description.text` (falling back to deprecated `annotations[]`)
- Post-render cleanup removes common webpage chrome (profile chips, comment UI controls, avatar markers, vote-counter lines)
- Removes alphabet-index navigation (runs of 5+ consecutive single-character blocks like A–Z filters)
- Cuts off known non-article sections (e.g., "More Articles from our Blog", "Community", "References")

### Memory Management

After every conversion, `torch.cuda.empty_cache()` and `gc.collect()` are called to prevent GPU memory leaks. All synchronous Docling operations run in threads via `asyncio.to_thread()` to avoid blocking FastAPI. A threading lock serializes converter access since the underlying models are not thread-safe.

## Enriched Document Fields

In addition to the filtered markdown, the proxy extracts three enriched fields from the raw Docling JSON document:

| Field | Description |
|-------|-------------|
| `title` | Page title from Docling's structured document — looks for `label="title"` text node in any content layer, falling back to first `section_header` |
| `hyperlinks` | All hyperlinks from the entire document (all content layers including navigation, headers, footers), resolved to absolute URLs |
| `content_hash` | SHA-256 hash of normalized markdown content (collapsed whitespace + lowercased). Used to skip duplicate content ingestion |

## Minimum Content Threshold

Documents are only stored in Qdrant if their cleaned `content` has at least `MIN_CONTENT_WORDS` words (default: `32`). Documents below the threshold are still scraped and returned to the caller, but embedding + Qdrant upsert is skipped and the response metadata includes `skipped_storage=true` and the computed `word_count`.

## Performance Optimizations

- **HNSW prefetch**: Fast approximate search before ColBERT rerank
- **Quantization rescoring**: Compressed vectors with precision recovery
- **Periodic memory cleanup**: Garbage collection every 30s
- **Background ingestion**: Web search results ingested asynchronously
- **Batch embedding**: Dense and ColBERT encoders process multiple documents per API call
- **vLLM serving**: Both ColBERT and Dense models served via vLLM for efficient GPU inference with continuous batching
