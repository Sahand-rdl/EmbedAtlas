"""
EmbedAtlas core — public API
Import from here rather than from submodules directly.
"""

from embedatlas.core.chunker import Chunk, Chunker
from embedatlas.core.embedder import Embedder, get_model_options, model_id_from_display
from embedatlas.core.exporter import (
    results_to_csv,
    results_to_json,
    sample_to_csv,
    sample_to_json,
)
from embedatlas.core.ingestion import (
    ingest_hf_dataset,
    ingest_local_file,
    ingest_local_folder,
    ingest_url,
)
from embedatlas.core.rag import RAGEngine, SearchResult
from embedatlas.core.reduction import compute_sample_size, reduce_and_plot
from embedatlas.core.vectorstore import CollectionInfo, VectorStore

__all__ = [
    "Chunk",
    "Chunker",
    "Embedder",
    "get_model_options",
    "model_id_from_display",
    "results_to_csv",
    "results_to_json",
    "sample_to_csv",
    "sample_to_json",
    "ingest_hf_dataset",
    "ingest_local_file",
    "ingest_local_folder",
    "ingest_url",
    "RAGEngine",
    "SearchResult",
    "compute_sample_size",
    "reduce_and_plot",
    "CollectionInfo",
    "VectorStore",
]
