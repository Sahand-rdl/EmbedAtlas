"""
EmbedAtlas — Central Configuration
All paths, constants, and model registries live here.
Nothing in this file has side effects; it is safe to import anywhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

# The *package* root (embedatlas/)
PACKAGE_ROOT = Path(__file__).parent.resolve()

# The *project* root (EmbedAtlas/) — one level up from the package
PROJECT_ROOT = PACKAGE_ROOT.parent.resolve()

# Where ChromaDB stores its persistent collections
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "collections"

# Scratch space for temporarily downloaded / extracted files
TEMP_DIR = PROJECT_ROOT / "data" / "tmp"

# ---------------------------------------------------------------------------
# ChromaDB defaults
# ---------------------------------------------------------------------------

CHROMA_DISTANCE_METRIC = "cosine"  # "cosine" | "l2" | "ip"
CHROMA_DEFAULT_COLLECTION = "my_collection"

# ---------------------------------------------------------------------------
# Chunking defaults  (overridable per session from the UI)
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 1000  # characters
DEFAULT_CHUNK_OVERLAP = 150  # characters
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 4000

# ---------------------------------------------------------------------------
# Embedding model registry
# Each entry:
#   "display_name": shown in the UI dropdown
#   "model_id":     passed to SentenceTransformer(...)
#   "description":  one-line tooltip shown in the UI
#   "max_seq_len":  tokens — used to warn the user about truncation
#   "dims":         output vector dimensions
#   "speed":        "fast" | "balanced" | "slow"   (for UI badge)
# ---------------------------------------------------------------------------

EMBEDDING_MODELS = [
    {
        "display_name": "all-MiniLM-L6-v2  [fast, 384-dim]",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Best for speed. Great general-purpose English model.",
        "max_seq_len": 256,
        "dims": 384,
        "speed": "fast",
    },
    {
        "display_name": "all-mpnet-base-v2  [balanced, 768-dim]",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "description": "High quality general English. Recommended default.",
        "max_seq_len": 384,
        "dims": 768,
        "speed": "balanced",
    },
    {
        "display_name": "BAAI/bge-small-en-v1.5  [fast, 384-dim]",
        "model_id": "BAAI/bge-small-en-v1.5",
        "description": "Excellent for RAG retrieval on English text.",
        "max_seq_len": 512,
        "dims": 384,
        "speed": "fast",
    },
    {
        "display_name": "BAAI/bge-m3  [multilingual, 1024-dim]",
        "model_id": "BAAI/bge-m3",
        "description": "State-of-the-art multilingual model. Slower but powerful.",
        "max_seq_len": 8192,
        "dims": 1024,
        "speed": "slow",
    },
    {
        "display_name": "paraphrase-multilingual-MiniLM-L12-v2  [multilingual, 384-dim]",
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "description": "Fast multilingual model. Good for 50+ languages.",
        "max_seq_len": 128,
        "dims": 384,
        "speed": "fast",
    },
    {
        "display_name": "all-MiniLM-L12-v2  [balanced, 384-dim]",
        "model_id": "sentence-transformers/all-MiniLM-L12-v2",
        "description": "Slightly better than L6 variant at modest speed cost.",
        "max_seq_len": 256,
        "dims": 384,
        "speed": "balanced",
    },
]

# Default model (index into EMBEDDING_MODELS)
DEFAULT_MODEL_INDEX = 1  # all-mpnet-base-v2

# ---------------------------------------------------------------------------
# Ingestion — supported local file extensions
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    "text": [".txt", ".md", ".rst"],
    "pdf": [".pdf"],
    "table": [".csv", ".tsv"],
    "json": [".json", ".jsonl"],
}

# Flat list for quick membership checks
ALL_SUPPORTED_EXTENSIONS: list[str] = [
    ext for exts in SUPPORTED_EXTENSIONS.values() for ext in exts
]

# ---------------------------------------------------------------------------
# HuggingFace ingestion
# ---------------------------------------------------------------------------

# Maximum rows to stream from an HF dataset in one session (safety cap)
# Users can override this in the UI up to HF_MAX_ROWS_HARD_CAP
HF_DEFAULT_MAX_ROWS = 10_000
HF_MAX_ROWS_HARD_CAP = 500_000

# Default text column name to look for when streaming HF datasets
HF_DEFAULT_TEXT_COLUMN = "text"

# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

# Hard upper limits beyond which the method becomes unstable / too slow
REDUCTION_MAX_POINTS = {
    "PCA": None,  # No limit — PCA is linear
    "UMAP": 50_000,
    "t-SNE": 10_000,
}

# If sample size > this fraction of the collection AND collection is large,
# switch to centroid-per-label strategy automatically
CENTROID_FALLBACK_FRACTION = 0.20  # 20 %
CENTROID_FALLBACK_MIN_DOCS = 50_000  # only applies when collection >= this

DEFAULT_SAMPLE_FRACTION = 0.05  # 5 % default sample for viz

# t-SNE defaults (mirroring your working scripts)
TSNE_DEFAULTS = {
    "perplexity": 30,
    "learning_rate": "auto",
    "init": "random",
    "random_state": 42,
    "max_iter": 500,
    "method": "barnes_hut",
    "angle": 0.5,
}

# UMAP defaults
UMAP_DEFAULTS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "random_state": 42,
}

# PCA defaults
PCA_DEFAULTS = {
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Search / RAG
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 10  # number of results returned
MAX_TOP_K = 100

# BM25 keyword results are normalized to [0,1] and multiplied by this weight
# when merging with semantic results in hybrid mode.
# Keyword hits always sort above pure-semantic hits regardless of this weight.
KEYWORD_SCORE_WEIGHT = 1.0
SEMANTIC_SCORE_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

# "pymupdf"  — fast, lightweight (default)
# "docling"  — layout-aware, heavy (~500 MB), optional install
DEFAULT_PDF_PARSER = "pymupdf"

# ---------------------------------------------------------------------------
# Misc UI
# ---------------------------------------------------------------------------

APP_TITLE = "EmbedAtlas"
APP_ICON = "🗺️"
APP_VERSION = "0.1.0"
