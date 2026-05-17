# 🗺️ EmbedAtlas

**The visual embedding workbench.**  
Ingest any text dataset → generate vector embeddings → explore clusters interactively → search with RAG. No coding required.

---

## What it does

| Step            | What you do                                                                       | What EmbedAtlas does                                                              |
| ------------------ | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **1 · Ingest**  | Upload files, point to a folder, paste a HuggingFace dataset ID, or drop in a URL | Loads and chunks your text with overlap-aware splitting                           |
| **2 · Embed**   | Pick an embedding model from the dropdown                                         | Encodes every chunk and stores vectors in ChromaDB                                |
| **3 · Explore** | Choose PCA, UMAP, or t-SNE                                                        | Renders an interactive Plotly scatter — hover any point to read the original text |
| **4 · Search**  | Type a query                                                                      | Returns ranked results via keyword, semantic, or hybrid search                    |

---

## Installation

```bash
pip install embedatlas
embedatlas
```

Your browser opens automatically at `http://localhost:8501`.

### GPU acceleration (optional)

EmbedAtlas works on CPU out of the box. For large datasets, a GPU dramatically speeds up embedding:

```bash
# NVIDIA (CUDA 12)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install embedatlas

# Apple Silicon (MPS)
pip install torch  # MPS is included in the standard macOS wheel
pip install embedatlas
```

### High-quality PDF parsing (optional)

The default PDF parser is PyMuPDF (fast, lightweight). For layout-aware parsing of tables and structured PDFs, install Docling:

```bash
pip install embedatlas[docling]
```

Then toggle "Docling" in the PDF parser option on the Ingest page.

---

## Supported data sources

| Source              | Examples                                                  |
| ------------------- | --------------------------------------------------------- |
| Local files         | `.txt` `.md` `.rst` `.pdf` `.csv` `.tsv` `.json` `.jsonl` |
| Local folder        | Any directory — EmbedAtlas recurses into sub-folders      |
| HuggingFace dataset | `allenai/c4`, `wikitext`, `imdb`, any public dataset repo |
| URL                 | Any direct link to a supported file type                  |

---

## Embedding models

EmbedAtlas ships with a curated selection of [SentenceTransformers](https://www.sbert.net/) models:

| Model                                   | Dims | Best for                               |
| --------------------------------------- | ---- | -------------------------------------- |
| `all-mpnet-base-v2`                     | 768  | High-quality general English (default) |
| `all-MiniLM-L6-v2`                      | 384  | Fast, great general English            |
| `BAAI/bge-small-en-v1.5`                | 384  | RAG retrieval on English               |
| `BAAI/bge-m3`                           | 1024 | Multilingual, state-of-the-art         |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384  | Fast multilingual (50+ languages)      |

Any model from the [SentenceTransformers model hub](https://huggingface.co/models?library=sentence-transformers) can be used by editing `config.py`.

---

## Collections

EmbedAtlas persists all embeddings in **ChromaDB** at `data/collections/`. Collections survive between sessions — open the app, pick up where you left off.

- **Create** a collection during Ingest
- **Switch** between collections from the sidebar
- **Delete** or **rename** collections via the sidebar settings panel

---

## Search modes

| Mode                       | How it works                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| **Hybrid** _(recommended)_ | Keyword hits ranked first, semantic results fill the rest. Each result shows a match badge. |
| **Semantic**               | Pure vector similarity — finds conceptually related chunks even without exact word matches  |
| **Keyword**                | BM25 term matching — fast, exact, no model required                                         |

---

## Tips

- **Always attach a label** during ingestion. Labels colour the scatter points in Explore and make clusters interpretable.
- **t-SNE** is best for ≤10k points. For larger collections, use UMAP or let EmbedAtlas switch to centroid-per-label mode automatically.
- **Hybrid search** is usually the right default. Use pure semantic search when your query is a concept or sentence rather than a specific term.
- For very large datasets, set a **max rows** limit on the HuggingFace tab to avoid OOM errors during embedding.

---

## Development

```bash
git clone https://github.com/your-org/embedatlas
cd embedatlas
pip install -e ".[dev]"
streamlit run embedatlas/app.py
```
