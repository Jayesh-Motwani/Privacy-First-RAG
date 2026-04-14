import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BENCHMARK_DIR = BASE_DIR / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"
GOLD_DATASET_PATH = BENCHMARK_DIR / "gold_dataset.json"
CONFLICT_MAP_PATH = BASE_DIR / "conflict_map.json"
CHROMA_DB_BASE_DIR = BASE_DIR / "chroma_db_benchmark"

# Models Under Study - LLMs (Local via Ollama)
LLM_MODELS = [
    "mistral:7b"
]

# Models Under Study - Embeddings (HuggingFace)
EMBEDDING_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # Current default
    "sentence-transformers/all-MiniLM-L6-v2",                      # Fast baseline
    "BAAI/bge-base-en-v1.5",                                       # Top MTEB performer
    "intfloat/multilingual-e5-base",                               # Strong multilingual
    "law-ai/InLegalBERT"                                           # Indian legal domain-specific
]

# Hyperparameters
TOP_K_RETRIEVAL = [3, 5, 10]  # Evaluate Recall@k, Precision@k at these thresholds

# Clustering Hyperparameters
CLUSTERING_CONFIG = {
    "HDBSCAN": {
        "min_cluster_size": 5,
        "metric": "euclidean"
    },
    "KMeans": {
        "k_values": [5, 10], # Compare different K values against HDBSCAN
        "n_init": 10
    },
    "UMAP_DIMENSIONS": 10,   # Pre-reduction dimension 5-15
    "UMAP_NEIGHBORS": 15
}

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_BASE_DIR, exist_ok=True)
