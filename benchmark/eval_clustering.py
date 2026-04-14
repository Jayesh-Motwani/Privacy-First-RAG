import json
import time
import umap
import hdbscan
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from benchmark.config import (
    EMBEDDING_MODELS, 
    CHROMA_DB_BASE_DIR, 
    CLUSTERING_CONFIG,
    RESULTS_DIR
)

def evaluate_clustering(embeddings, labels, algo_name):
    """Computes basic clustering metrics, accommodating HDBSCAN's noise (-1) logic."""
    noise_ratio = 0.0
    labels_for_metrics = labels
    valid_embeddings = embeddings
    
    if algo_name == "HDBSCAN":
        noise_ratio = np.sum(labels == -1) / len(labels)
        
        # Silhouette and DB scores require > 1 cluster and aren't well defined on noise points
        valid_indices = labels != -1
        if len(set(labels[valid_indices])) > 1:
            labels_for_metrics = labels[valid_indices]
            valid_embeddings = embeddings[valid_indices]
        else:
            return -1.0, -1.0, noise_ratio
            
    try:
        sil_score = silhouette_score(valid_embeddings, labels_for_metrics)
        db_score = davies_bouldin_score(valid_embeddings, labels_for_metrics)
    except ValueError:
        sil_score = -1.0
        db_score = -1.0
        
    return sil_score, db_score, noise_ratio

def run_clustering_benchmark():
    print(f"Starting Semantic Clustering Evaluation Matrix (HDBSCAN vs KMeans)...\n")
    results = {}
    
    for embedder_name in EMBEDDING_MODELS:
        print(f"[{embedder_name}] Loading raw embedded chunks...")
        db_folder_name = embedder_name.replace("/", "_")
        persist_dir = CHROMA_DB_BASE_DIR / db_folder_name
        
        if not persist_dir.exists():
            print(f"    [Skip] {embedder_name} not ingested yet. Run eval_retrieval.py first.")
            continue
            
        embeddings_model = HuggingFaceEmbeddings(
            model_name=embedder_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = Chroma(
            persist_directory=str(persist_dir), 
            embedding_function=embeddings_model,
            collection_name="legal_documents"
        )
        
        # Get all chunk embeddings for clustering
        all_data = vectorstore.get(include=["embeddings"])
        embeddings = np.array(all_data["embeddings"])
        chunks_count = len(embeddings)
        print(f"    Extracted {chunks_count} chunks. Dimension: {embeddings.shape[1]}")
        
        start_umap = time.time()
        # 1. Pre-process explicitly as per plan
        reducer = umap.UMAP(
            n_neighbors=CLUSTERING_CONFIG["UMAP_NEIGHBORS"], 
            n_components=CLUSTERING_CONFIG["UMAP_DIMENSIONS"], 
            metric='cosine', 
            random_state=42
        )
        umap_embeddings = reducer.fit_transform(embeddings)
        print(f"    Reduced dimensions linearly via UMAP in {time.time() - start_umap:.2f}s")
        
        embedder_results = {}
        
        # 2A. Run HDBSCAN
        print(f"    -> Running HDBSCAN...")
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=CLUSTERING_CONFIG["HDBSCAN"]["min_cluster_size"],
            metric=CLUSTERING_CONFIG["HDBSCAN"]["metric"]
        )
        t0 = time.time()
        hdb_labels = hdb.fit_predict(umap_embeddings)
        t_hdb = time.time() - t0
        
        num_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        sil_hdb, db_hdb, hdb_noise = evaluate_clustering(umap_embeddings, hdb_labels, "HDBSCAN")
        
        embedder_results["HDBSCAN"] = {
            "num_clusters_found": num_clusters_hdb,
            "silhouette_score": round(sil_hdb, 4),
            "davies_bouldin": round(db_hdb, 4),
            "noise_ratio": round(hdb_noise, 4),
            "latency_sec": round(t_hdb, 4)
        }
        
        # 2B. Run K-Means baselines
        for k in CLUSTERING_CONFIG["KMeans"]["k_values"]:
            print(f"    -> Running K-Means (k={k})...")
            kmeans = KMeans(n_clusters=k, n_init=CLUSTERING_CONFIG["KMeans"]["n_init"], random_state=42)
            
            t0 = time.time()
            km_labels = kmeans.fit_predict(umap_embeddings)
            t_km = time.time() - t0
            
            sil_km, db_km, km_noise = evaluate_clustering(umap_embeddings, km_labels, "KMeans")
            
            embedder_results[f"KMeans_k={k}"] = {
                "num_clusters_found": k,
                "silhouette_score": round(sil_km, 4),
                "davies_bouldin": round(db_km, 4),
                "noise_ratio": round(km_noise, 4),
                "latency_sec": round(t_km, 4)
            }
            
        results[embedder_name] = embedder_results
        
    out_path = RESULTS_DIR / "clustering_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[Done] Clustering evaluation matrices dumped to {out_path}")

if __name__ == "__main__":
    run_clustering_benchmark()
