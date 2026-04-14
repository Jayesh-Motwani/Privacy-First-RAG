import json
import time
import psutil
from pathlib import Path
import numpy as np

# Change directory context so local modules load
import sys
import os
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from langchain_huggingface import HuggingFaceEmbeddings
from ingest import run_ingestion
from benchmark.config import (
    GOLD_DATASET_PATH, 
    EMBEDDING_MODELS, 
    CHROMA_DB_BASE_DIR, 
    DATA_DIR, 
    TOP_K_RETRIEVAL,
    RESULTS_DIR
)
from langchain_chroma import Chroma

def compute_mrr(retrieved_docs, gold_passages):
    for rank, doc in enumerate(retrieved_docs, 1):
        for gold in gold_passages:
            if doc.metadata.get('act_name') == gold.get('act_name') and \
               doc.metadata.get('section_number') == gold.get('section_number'):
                return 1.0 / rank
    return 0.0

def compute_recall_at_k(retrieved_docs, gold_passages, k):
    retrieved_k = retrieved_docs[:k]
    matches = 0
    for gold in gold_passages:
        for doc in retrieved_k:
            if doc.metadata.get('act_name') == gold.get('act_name') and \
               doc.metadata.get('section_number') == gold.get('section_number'):
                matches += 1
                break
    return matches / len(gold_passages) if gold_passages else 0.0

def compute_precision_at_k(retrieved_docs, gold_passages, k):
    retrieved_k = retrieved_docs[:k]
    matches = 0
    for doc in retrieved_k:
        is_match = False
        for gold in gold_passages:
            if doc.metadata.get('act_name') == gold.get('act_name') and \
               doc.metadata.get('section_number') == gold.get('section_number'):
                is_match = True
                break
        if is_match:
            matches += 1
    return matches / k if k > 0 else 0.0

def run_retrieval_benchmark():
    print(f"Loading Gold Dataset from {GOLD_DATASET_PATH}...")
    with open(GOLD_DATASET_PATH, 'r') as f:
        gold_dataset = json.load(f)
        
    results = {}

    for embedder_name in EMBEDDING_MODELS:
        print(f"\n{'='*50}\nEvaluating Embedder: {embedder_name}\n{'='*50}")
        
        # 1. Setup specific Chroma DB path for this embedder
        db_folder_name = embedder_name.replace("/", "_")
        persist_dir = CHROMA_DB_BASE_DIR / db_folder_name
        
        # 2. Re-ingest the documents into this specific vectorstore using the given embedder
        print(f"[{embedder_name}] Step 1: Re-ingesting documents...")
        start_ingest_time = time.time()
        
        # To strictly avoid wiping main db, ensure we pass the newly mapped persist dir
        run_ingestion(
            data_dir=str(DATA_DIR), 
            clear_first=True, 
            persist_directory=str(persist_dir), 
            embedding_model=embedder_name
        )
        ingest_time = time.time() - start_ingest_time
        
        # Capture storage size
        storage_size_bytes = sum(f.stat().st_size for f in persist_dir.glob('**/*') if f.is_file())
        storage_size_mb = storage_size_bytes / (1024 * 1024)

        # 3. Initialize Chroma directly to skip pipeline wrapper for pure precision-retrieval
        print(f"[{embedder_name}] Step 2: Running 100 queries...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedder_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = Chroma(
            persist_directory=str(persist_dir), 
            embedding_function=embeddings,
            collection_name="legal_documents"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": max(TOP_K_RETRIEVAL)})
        
        embedder_metrics = {
            "MRR": [],
            "ingestion_time_sec": ingest_time,
            "storage_size_mb": storage_size_mb,
            "avg_embedding_latency_ms": 0
        }
        for k in TOP_K_RETRIEVAL:
            embedder_metrics[f"Recall@{k}"] = []
            embedder_metrics[f"Precision@{k}"] = []
            
        latency_measurements = []
        
        # 4. Iterate queries
        for q in gold_dataset:
            # We enforce testing applicability filtering natively via metadata here
            # but since vanilla retriever is tested, we'll test bare semantic retrieval
            # then filtering drop-outs next in ablation.
            
            q_text = q['expected_rewritten_query']  # Use rewritten as it represents the theoretical 'perfect' engine query passing context
            gold_passages = q['gold_passages']
            
            start_q_time = time.time()
            retrieved_docs = retriever.invoke(q_text)
            latency_measurements.append((time.time() - start_q_time) * 1000)
            
            embedder_metrics["MRR"].append(compute_mrr(retrieved_docs, gold_passages))
            
            for k in TOP_K_RETRIEVAL:
                embedder_metrics[f"Recall@{k}"].append(compute_recall_at_k(retrieved_docs, gold_passages, k))
                embedder_metrics[f"Precision@{k}"].append(compute_precision_at_k(retrieved_docs, gold_passages, k))
                
        # 5. Aggregate averages
        final_metrics = {
            "ingestion_time_sec": round(ingest_time, 2),
            "storage_size_mb": round(storage_size_mb, 2),
            "avg_embedding_latency_ms": round(np.mean(latency_measurements), 2),
            "MRR": round(np.mean(embedder_metrics["MRR"]), 4)
        }
        for k in TOP_K_RETRIEVAL:
            final_metrics[f"Recall@{k}"] = round(np.mean(embedder_metrics[f"Recall@{k}"]), 4)
            final_metrics[f"Precision@{k}"] = round(np.mean(embedder_metrics[f"Precision@{k}"]), 4)
            
        results[embedder_name] = final_metrics
        print(f"Results for {embedder_name}:", final_metrics)

    # Save to disk
    out_path = RESULTS_DIR / "retrieval_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[Done] All retrieval baseline metrics written to {out_path}")

if __name__ == "__main__":
    run_retrieval_benchmark()
