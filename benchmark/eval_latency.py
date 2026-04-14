import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from benchmark.config import RESULTS_DIR

def run_latency_analysis():
    print("Aggregating Latency Profiles from Component Evaluations...")
    
    latency_summary = {
        "pipeline_stages": {},
        "end_to_end_llm_configs": {},
        "embedding_configs": {},
        "clustering_configs": {}
    }
    
    # 1. End-To-End Pipeline Latency
    e2e_path = RESULTS_DIR / "e2e_results.json"
    if e2e_path.exists():
        with open(e2e_path, 'r') as f:
            e2e_data = json.load(f)
            for llm, metrics in e2e_data.items():
                latency_summary["end_to_end_llm_configs"][llm] = metrics.get("avg_pipeline_latency_sec", 0.0)
                
    # 2. Embedding Latencies
    retrieval_path = RESULTS_DIR / "retrieval_results.json"
    if retrieval_path.exists():
        with open(retrieval_path, 'r') as f:
            retrieval_data = json.load(f)
            for embedder, metrics in retrieval_data.items():
                latency_summary["embedding_configs"][embedder] = metrics.get("avg_embedding_latency_ms", 0.0)
                
    # 3. Rewriting Latencies
    rewriting_path = RESULTS_DIR / "rewriting_results.json"
    if rewriting_path.exists():
        with open(rewriting_path, 'r') as f:
            rewriting_data = json.load(f)
            for llm, metrics in rewriting_data.items():
                latency_summary["pipeline_stages"][f"Query_Rewriting_{llm}"] = metrics.get("avg_latency_ms", 0.0)
                
    # 4. Conflict Detection Latencies
    conflict_path = RESULTS_DIR / "conflict_results.json"
    if conflict_path.exists():
        with open(conflict_path, 'r') as f:
            conflict_data = json.load(f)
            latency_summary["pipeline_stages"]["Conflict_Detection"] = conflict_data.get("avg_detection_latency_ms", 0.0)

    # 5. Clustering Latencies
    clustering_path = RESULTS_DIR / "clustering_results.json"
    if clustering_path.exists():
        with open(clustering_path, 'r') as f:
            clustering_data = json.load(f)
            for embedder, metrics in clustering_data.items():
                for algo, stats in metrics.items():
                    latency_summary["clustering_configs"][f"{embedder}::{algo}"] = stats.get("latency_sec", 0.0)

    out_path = RESULTS_DIR / "latency_results.json"
    with open(out_path, 'w') as f:
        json.dump(latency_summary, f, indent=4)
        
    print(f"Latency Profile Summary successfully aggregated and dumped to {out_path}")

if __name__ == "__main__":
    run_latency_analysis()
