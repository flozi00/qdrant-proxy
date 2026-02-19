"""
Static test fixtures for qdrant-proxy MCP endpoint tests.

The document content below is a snapshot of the Wikipedia "Vector database"
article fetched on 2026-02-19. It is stored as a constant so that production
assertion strings are deterministic and independent of any live web fetch.
"""

# ---------------------------------------------------------------------------
# Primary test document – Wikipedia: Vector database
# ---------------------------------------------------------------------------

TEST_PRIMARY_URL = "https://en.wikipedia.org/wiki/Vector_database"

# Phrases that MUST appear in the indexed document content.
# These are used to assert that search returns the correct document.
EXPECTED_PHRASES = [
    "approximate nearest neighbor",
    "retrieval-augmented generation",
    "Hierarchical Navigable Small World",
    "Qdrant",
    "similarity search",
]

# Static content snapshot (plain text / markdown, free of wiki markup).
# Trimmed to the essential sections to keep the fixture manageable.
TEST_DOCUMENT_CONTENT = """\
# Vector database

A vector database, vector store or vector search engine is a database that
stores and retrieves embeddings of data in vector space. Vector databases
typically implement approximate nearest neighbor algorithms so users can
search for records semantically similar to a given input, unlike traditional
databases which primarily look up records by exact match. Use-cases for vector
databases include similarity search, semantic search, multi-modal search,
recommendations engines, object detection, and retrieval-augmented generation
(RAG).

Vector embeddings are mathematical representations of data in a high-dimensional
space. In this space, each dimension corresponds to a feature of the data, with
the number of dimensions ranging from a few hundred to tens of thousands,
depending on the complexity of the data being represented. Each data item is
represented by one vector in this space. Words, phrases, or entire documents,
as well as images, audio, and other types of data, can all be vectorized.

These feature vectors may be computed from the raw data using machine learning
methods such as feature extraction algorithms, word embeddings or deep learning
networks. The goal is that semantically similar data items receive feature
vectors close to each other.

## Techniques

The most important techniques for similarity search on high-dimensional vectors
include:

- Hierarchical Navigable Small World (HNSW) graphs
- Locality-sensitive Hashing (LSH) and Sketching
- Product Quantization (PQ)
- Inverted Files

and combinations of these techniques. In recent benchmarks, HNSW-based
implementations have been among the best performers.

## Applications

Vector databases are used in a wide range of machine learning applications
including similarity search, semantic search, multi-modal search,
recommendations engines, object detection, and retrieval-augmented generation.

### Retrieval-augmented generation

An especially common use-case for vector databases is in
retrieval-augmented generation (RAG), a method to improve domain-specific
responses of large language models. The retrieval component of a RAG can be
any search system, but is most often implemented as a vector database. Text
documents describing the domain of interest are collected, and for each
document or document section, a feature vector (known as an "embedding") is
computed, typically using a deep learning network, and stored in a vector
database along with a link to the document. Given a user prompt, the feature
vector of the prompt is computed, and the database is queried to retrieve the
most relevant documents. These are then automatically added into the context
window of the large language model, and the large language model proceeds to
create a response to the prompt given this context.

## Implementations

Notable open-source and proprietary vector database implementations include:

- Qdrant (Apache License 2.0)
- Milvus (Apache License 2.0)
- Weaviate (BSD 3-Clause)
- Chroma (Apache License 2.0)
- Elasticsearch
- Pinecone (Proprietary Managed Service)
- Postgres with pgvector
- Redis Stack
- LanceDB (Apache License 2.0)
"""

TEST_DOCUMENT_TITLE = "Vector database - Wikipedia"

# ---------------------------------------------------------------------------
# Secondary test document – used for add_source_to_faq_entry
# ---------------------------------------------------------------------------

TEST_SECONDARY_URL = "https://en.wikipedia.org/wiki/Approximate_nearest_neighbor_search"

# ---------------------------------------------------------------------------
# KV / FAQ store test collection name
# Produces Qdrant collection: kv_mcp_test_suite
# ---------------------------------------------------------------------------

KV_TEST_COLLECTION = "mcp-test-suite"

# ---------------------------------------------------------------------------
# Test FAQ data
# ---------------------------------------------------------------------------

TEST_FAQ_QUESTION = "What is a vector database?"
TEST_FAQ_ANSWER = (
    "A vector database is a database that stores and retrieves embeddings of "
    "data in vector space, implementing approximate nearest neighbor algorithms "
    "for semantic similarity search."
)

# KV entry
KV_TEST_KEY = "What search algorithms do vector databases use?"
KV_TEST_VALUE = (
    "Vector databases primarily use Hierarchical Navigable Small World (HNSW) "
    "graphs, Locality-sensitive Hashing (LSH), and Product Quantization (PQ) "
    "for approximate nearest neighbor search."
)
