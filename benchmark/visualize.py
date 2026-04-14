import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_retrieval():
    path = RESULTS_DIR / "retrieval_results.json"
    if not path.exists(): return
    with open(path) as f:
        data = json.load(f)
        
    models = list(data.keys())
    rc5 = [data[m]["Recall@5"] for m in models]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rc5, y=models, palette="viridis")
    plt.title("Fig 2: Retrieval Recall@5 across 5 Embedding Models", pad=15)
    plt.xlabel("Recall@5")
    plt.ylabel("Embedder")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig2_Retrieval_Recall_k5.png")
    plt.close()

def plot_clustering():
    path = RESULTS_DIR / "clustering_results.json"
    if not path.exists(): return
    with open(path) as f:
        data = json.load(f)
        
    embedder = list(data.keys())[0] # Pick the first
    results = data[embedder]
    
    algos = list(results.keys())
    noise = [results[algo].get("noise_ratio", 0) for algo in algos]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=algos, y=noise, palette="rocket")
    plt.title(f"HDBSCAN vs K-means: Outlier Noise Extraction ({embedder.split('/')[-1]})", pad=15)
    plt.ylabel("Noise Ratio (% chunks assigned to -1)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Clustering_Noise_Ratio.png")
    plt.close()

def plot_ablation():
    path = RESULTS_DIR / "ablation_results.json"
    if not path.exists(): return
    with open(path) as f:
        data = json.load(f)
        
    modes = list(data.keys())
    scores = [data[m]["completeness_score"] for m in modes]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=modes, palette="magma")
    plt.title("Fig 6: Component Ablation (Completeness Score)", pad=15)
    plt.xlabel("Completeness Score Proxy")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig6_Component_Ablation.png")
    plt.close()

def plot_latency():
    path = RESULTS_DIR / "latency_results.json"
    if not path.exists(): return
    with open(path) as f:
        data = json.load(f)
        
    e2e = data.get("end_to_end_llm_configs", {})
    if not e2e: return
    
    models = list(e2e.keys())
    latencies = [e2e[m] for m in models]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=latencies, y=models, palette="mako")
    plt.title("Fig 7: Pipeline Latency (End-to-End)", pad=15)
    plt.xlabel("Latency (Seconds)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig7_Latency_Distribution.png")
    plt.close()

def main():
    print("Generating visualizations...")
    plot_retrieval()
    plot_clustering()
    plot_ablation()
    plot_latency()
    print(f"Figures saved successfully to {FIG_DIR}")

if __name__ == "__main__":
    main()
