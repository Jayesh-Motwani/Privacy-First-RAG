import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import subprocess

def run_script(script_name):
    print(f"\n=======================================================")
    print(f"Executing: {script_name}")
    print(f"=======================================================")
    
    script_path = BASE_DIR / "benchmark" / script_name
    result = subprocess.run([sys.executable, str(script_path)])
    
    if result.returncode != 0:
        print(f"\n[ERROR] execution of {script_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    print("Starting Full Privacy-First-RAG Evaluation Suite...")
    
    # Pipeline execution order according to benchPlan Component Ablations & Tiers
    scripts = [
        "eval_retrieval.py",   # Re-ingests all data conditionally per embedder and runs semantic Recall@K
        "eval_rewriting.py",   # LLM latency and Query Semantic similarity
        "eval_conflict.py",    # Rule-based precision recall for tensions
        "eval_clustering.py",  # HDBSCAN vs K-means purity tests
        "eval_e2e.py",         # RAGAS tier generation capabilities
        "eval_ablation.py",    # Structural Pipeline Disablement checks
        "eval_latency.py",     # Component resource and latency metrics
        "visualize.py"         # Output table visualizations
    ]
    
    for s in scripts:
        run_script(s)
        
    print("\n---------------------------------------------------------")
    print("All benchmark evaluations completed successfully!")
    print("Outputs stored in: ./benchmark/results/")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    main()
