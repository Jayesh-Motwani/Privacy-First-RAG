# Benchmarking Strategy: Privacy-First Legal RAG Pipeline
## Research Paper — Full Solo-Executable Experimental Protocol

---

## 1. Research Paper Framing & Novelty Claims

Your paper's unique contributions (what reviewers care about):

| # | Novelty Claim | Status | Benchmark Needed |
|---|--------------|--------|-----------------|
| N1 | **Hierarchical Legal Document Parsing** — Custom parser for Indian statute structure (Act→Part→Chapter→Section→Proviso) | ✅ Implemented | Chunk quality, retrieval recall vs. naive chunking |
| N2 | **Conflict-Aware RAG** — Runtime detection of tensions between legal provisions | ✅ Implemented | Conflict detection precision/recall, answer quality impact |
| N3 | **Applicability Filtering** — Metadata-based filtering by personal law, jurisdiction, demographic | ✅ Implemented | Filtering accuracy, retrieval precision improvement |
| N4 | **100% Local, Zero-Cost Pipeline** — All models free & local via Ollama + HuggingFace | ✅ Implemented | Latency, resource usage, accuracy vs. cloud baselines |

> [!IMPORTANT]
> The paper should be positioned at the intersection of **Legal NLP**, **Privacy-Preserving AI**, and **RAG Systems**. Target venues: ACL Workshop on NLP and Law, EMNLP Industry Track, CIKM Applied Track, or ACM CS&Law symposium.

---

## 2. Gold-Standard Dataset Construction (YOU Do This Yourself)

This is the most critical step. You need a **test set of question-answer pairs** grounded in your ingested legal corpus. No lawyers needed — your corpus is **public Indian legislation** and you can read the statutes yourself.

### 2.1 Dataset Design

Create **100 question-answer pairs** across these dimensions:

| Dimension | Categories | # Questions Each |
|-----------|-----------|-----------------|
| **Query Complexity** | Simple (single-section lookup), Complex (multi-section reasoning), Cross-Act (requires provisions from 2+ Acts) | 33 / 34 / 33 |
| **Legal Domain** | Domestic Violence, Child Protection (POCSO/JJ), Marriage & Divorce (HMA), Maintenance (CrPC S.125 / HAMA), Adoption (HAMA/JJ) | 20 each |
| **Conflict Scenarios** | Queries that should trigger conflict detection, Queries with no conflicts | 25 / 75 |
| **Demographics** | married_woman, minor, divorced_woman, any | 25 each |

### 2.2 Annotation Format (JSON)

```json
{
  "id": "Q001",
  "raw_query": "My husband Ramesh beats me daily. What protection can I get?",
  "expected_rewritten_query": "What remedies are available under the Protection of Women from Domestic Violence Act, 2005 for physical abuse by husband?",
  "gold_passages": [
    {
      "act_name": "Protection of Women from Domestic Violence Act, 2005",
      "section_number": "18",
      "section_title": "Protection orders",
      "passage_text": "The Magistrate may, after giving the aggrieved person..."
    },
    {
      "act_name": "Protection of Women from Domestic Violence Act, 2005",
      "section_number": "23",
      "section_title": "Power to grant interim and ex parte orders"
    }
  ],
  "gold_answer_key_points": [
    "DV Act Section 18 protection orders",
    "Section 23 interim/ex parte orders available",
    "Section 19 residence orders",
    "Can file complaint under Section 12"
  ],
  "expected_conflicts": [],
  "user_profile": {
    "jurisdiction": "central",
    "personal_law": "hindu",
    "demographic": "married_woman"
  },
  "complexity": "complex",
  "domain": "domestic_violence"
}
```

### 2.3 How to Create the 100 QA Pairs (Step-by-Step)

1. **Open each PDF** in `data/` and read through the key sections
2. **For each Act**, write 15-20 natural-language questions a real person would ask
3. **For the gold passages**, copy the exact section text from the PDF
4. **For key points**, list 3-5 factual claims the answer MUST contain
5. **For conflict scenarios**, pair questions that touch provisions listed in `conflict_map.json`

> [!TIP]
> Start with 50 questions. Run the pipeline. See where it fails. Write 50 more questions targeting failure modes. This gives you a naturally balanced dataset.

### 2.4 Save Location

```
Privacy-First-RAG/
├── benchmark/
│   ├── gold_dataset.json          # 100 QA pairs
│   ├── gold_dataset_mini.json     # 20 QA pairs for quick iteration
│   └── annotation_guidelines.md   # Your annotation rules (for reproducibility section)
```

---

## 3. Evaluation Framework — 4 Tiers

### Tier 1: Component-Level Evaluation (Isolate Each Module)

#### 3.1 Query Rewriting Evaluation (`LegalQueryRewriter`)

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Legal Term Recall** | % of expected legal terms present in rewritten query | Check for act names, section references, legal vocabulary from gold |
| **Semantic Similarity** | How close is rewritten query to gold rewritten query | Cosine similarity of embeddings |
| **Query Expansion Rate** | How much richer is the rewritten query | `len(rewritten_tokens) / len(original_tokens)` |
| **Hallucination Rate** | Does rewriter inject wrong Act/Section references | Manual check on 50 queries |

#### 3.2 Retrieval Evaluation (ChromaDB + Embeddings)

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Recall@k** (k=3,5,10) | % of gold passages retrieved in top-k | Match retrieved chunks against `gold_passages` |
| **Precision@k** | % of retrieved passages that are in gold set | Standard IR precision |
| **MRR** (Mean Reciprocal Rank) | Rank of the first relevant passage | `1/rank_of_first_relevant` |
| **nDCG@k** | Quality of ranking considering all relevant docs | Standard nDCG computation |
| **Context Coverage** | % of gold key points answerable from retrieved context | Manual or LLM-judged per key point |

**This is your MOST important evaluation** — as Legal RAG Bench (2603.01710) shows, retrieval sets the ceiling for end-to-end performance.

#### 3.3 Semantic Clustering Evaluation

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Silhouette Score** | Density and separation of clusters formed | `sklearn.metrics.silhouette_score` |
| **Davies-Bouldin Index** | Similarity measure of each cluster with its most similar cluster | `sklearn.metrics.davies_bouldin_score` |
| **Noise Ratio** | % of chunks flagged as outlier/noise | `len(labels == -1) / len(chunks)` |
| **Cluster-Guided Recall** | Retrieval recall when pre-filtering via cluster topics | Rerun Tier 1 Retrieval pipeline conditionally |

#### 3.4 Conflict Detection Evaluation

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Conflict Precision** | % of flagged conflicts that are real | `TP / (TP + FP)` against gold conflicts |
| **Conflict Recall** | % of real conflicts detected | `TP / (TP + FN)` against gold conflicts |
| **False Alarm Rate** | How often system incorrectly flags conflicts | On non-conflict queries |

---

### Tier 2: End-to-End RAG Evaluation (RAGAS-Style)

Use the [RAGAS](https://docs.ragas.io/) framework metrics, computed using one of your local LLMs as the judge:

| Metric | What It Measures | Requires |
|--------|-----------------|----------|
| **Faithfulness** | Is the answer grounded in retrieved context only? | Retrieved context + answer |
| **Answer Relevancy** | Does the answer address the actual question? | Question + answer |
| **Context Precision** | Are the most relevant docs ranked highest? | Question + retrieved context + gold passages |
| **Context Recall** | Does retrieved context cover all needed info? | Retrieved context + gold answer |

**Implementation:**
```python
# benchmark/eval_e2e.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper

# Use a LOCAL LLM as judge (e.g., qwen2.5:14b or mistral:7b)
judge_llm = LangchainLLMWrapper(OllamaLLM(model="qwen2.5:14b"))

results = evaluate(
    dataset=your_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm
)
```

> [!NOTE]
> Using a local LLM as RAGAS judge is acceptable for a research paper. Just be transparent about it. Mention the judge model and its limitations in Section 5 (Limitations).

### Tier 3: Legal-Domain-Specific Metrics

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Citation Accuracy** | Does the answer cite the correct Act + Section numbers? | Extract section references from answer, match against gold |
| **Legal Completeness** | Does the answer cover all relevant provisions? | `key_points_covered / total_key_points` from gold |
| **Conflict Mention Rate** | When conflicts exist, does the answer mention them? | Check answer text for conflict discussion |
| **Applicability Correctness** | Are retrieved provisions applicable to the user's profile? | Verify metadata (personal_law, demographic) matches profile |

### Tier 4: Efficiency Metrics

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Pipeline Latency** | Total time from query to answer | `time.time()` around `process_query()` |
| **Component Latency** | Time per stage (rewrite, retrieve, generate) | Timer per step |
| **Memory Usage** | Peak RAM during query processing | `psutil` / `tracemalloc` |
| **GPU/CPU Utilization** | Resource consumption | System monitoring |

---

## 4. Ablation Study Design

### 4.1 LLM Ablation (Query Rewriting + Answer Generation)

Test these **5 local LLMs** via Ollama (all free, all run on your machine):

| Model | Parameters | Why Include |
|-------|-----------|-------------|
| `qwen3.5:4b` | 4B | Your current default; strong multilingual |
| `mistral:7b` | 7B | Your alternative; efficient and fast |
| `llama3.2:3b` | 3B | Meta's flagship; strong general reasoning |
| `gemma4:e4b` | 4B | Google's latest; strong instruction following |


#### LLM Experiment Matrix

```
For each LLM in [qwen2.5:7b, mistral:7b, llama3.2:8b, phi4:14b, gemma3:12b]:
    For each task in [query_rewriting, answer_generation]:
        Run all 100 queries
        Compute all Tier 1-4 metrics
        Record latency, memory
```

**Total runs**: 5 LLMs × 2 tasks × 100 queries = **1,000 inference runs**

### 4.2 Embedding Model Ablation (Retrieval Quality)

Test these **5 embedding models** (all free via HuggingFace):

| Model | Dimensions | Why Include |
|-------|-----------|-------------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Your current default; lightweight multilingual |
| `all-MiniLM-L6-v2` | 384 | Popular English-only baseline; fastest |
| `BAAI/bge-base-en-v1.5` | 768 | Top MTEB performer; strong on retrieval |
| `intfloat/multilingual-e5-base` | 768 | Strong multilingual alternative |
| `law-ai/InLegalBERT` | 768 | Indian legal domain-specific BERT |

#### Embedding Experiment Matrix

```
For each embedder in [MiniLM-multi, MiniLM-L6, BGE-base, E5-multi, InLegalBERT]:
    1. Re-ingest all documents with this embedder
    2. Run all 100 queries (retrieval only, no generation)
    3. Compute Recall@k, Precision@k, MRR, nDCG@k
    4. Record ingestion time, embedding latency, storage size
```

**Total runs**: 5 embedders × 100 queries = **500 retrieval runs** (plus 5 re-ingestions)

### 4.3 Clustering Strategy (HDBSCAN vs. K-Means)

Compare unsupervised chunk grouping mechanisms (used for topic-aware routing or hierarchical grouping):

| Algorithm | Why Include |
|-----------|-------------|
| **HDBSCAN** | Density-based model. Can discover cluster count dynamically and isolate outlier chunks (`-1` noise) from legal topics without degrading clusters. |
| **K-Means** | Centroid-based default model. Forces all chunks into $k$ spherical clusters. |

#### Clustering Experiment Matrix

```
For each algorithm in [HDBSCAN (min_size=5), K-Means (k=5), K-Means (k=10)]:
    1. Pre-process embeddings via UMAP (dim=5-15)
    2. Cluster chunks from the primary vectorstore
    3. Compute Silhouette Score & Davies-Bouldin Index
    4. Calculate retrieval precision when filtering via clusters
```

### 4.4 Full Factorial Cross-Ablation (Key Contribution)

Pick your **top 3 LLMs** and **top 3 embedders** from 4.1/4.2 results, then run the full pipeline:

```
3 LLMs × 3 Embedders × 100 queries = 900 full pipeline runs
```

This produces a **3×3 heatmap** — the centerpiece figure of your paper.

### 4.5 Component Ablation (What Happens When You Remove Each Novelty?)

| Experiment | What You Disable | What You Measure |
|-----------|-----------------|-----------------|
| **No Query Rewriting** | Send raw query directly to retrieval | Retrieval recall drop, answer quality drop |
| **No Conflict Detection** | Skip `ConflictDetector` | Answer quality on conflict scenarios only |
| **No Applicability Filtering** | Remove metadata filters from ChromaDB | Retrieval precision drop, irrelevant results |
| **Naive Chunking** | Replace hierarchical parser with fixed-size 512-token chunks | Retrieval recall/precision change |
| **Full Pipeline** | Everything enabled (baseline) | Compare all above against this |

> [!IMPORTANT]
> This component ablation is what makes the paper strong. It proves each module adds measurable value. Reviewers love this.

---

## 5. Experimental Protocol (Step-by-Step)

### Phase 1: Dataset Creation (Estimated: 2-3 days)

```
Day 1: Read through all 8 PDFs, annotate 50 QA pairs
Day 2: Write 50 more QA pairs targeting edge cases and conflict scenarios
Day 3: Review, clean, and validate the dataset; create mini subset
```

**Deliverable**: `benchmark/gold_dataset.json` (100 QA pairs)

### Phase 2: Benchmark Harness Implementation (Estimated: 3-4 days)

Build the automation scripts:

```
benchmark/
├── config.py                # Model lists, paths, hyperparameters
├── eval_retrieval.py        # Retrieval metrics (Recall@k, Precision@k, MRR, nDCG)
├── eval_clustering.py       # Cluster overlap, Silhouette score, UMAP viz (HDBSCAN vs KMeans)
├── eval_rewriting.py        # Query rewriting evaluation
├── eval_conflict.py         # Conflict detection precision/recall
├── eval_e2e.py              # Full pipeline RAGAS metrics
├── eval_ablation.py         # Component ablation runner
├── eval_latency.py          # Latency and resource usage
├── run_all.py               # Master script to run everything
├── results/                 # Output directory
│   ├── retrieval_results.json
│   ├── clustering_results.json
│   ├── e2e_results.json
│   ├── ablation_results.json
│   └── latency_results.json
└── visualize.py             # Generate tables and plots for paper
```

### Phase 3: Run Experiments (Estimated: 3-5 days)

```
Step 1: Pull all Ollama models
  ollama pull qwen2.5:7b
  ollama pull mistral:7b
  ollama pull llama3.2:8b
  ollama pull phi4:14b
  ollama pull gemma3:12b

Step 2: Run embedding ablation (re-ingest with each embedder)
  python benchmark/run_all.py --phase embeddings

Step 3: Run LLM ablation (test each LLM for rewriting + generation)
  python benchmark/run_all.py --phase llm

Step 4: Run component ablation (disable each feature in turn)
  python benchmark/run_all.py --phase components

Step 5: Run full factorial cross-ablation (top 3 × top 3)
  python benchmark/run_all.py --phase cross

Step 6: Run privacy and latency benchmarks
  python benchmark/run_all.py --phase privacy_latency
```

### Phase 4: Analysis & Visualization (Estimated: 2 days)

Generate these figures for the paper:

| Figure | Type | Content |
|--------|------|---------|
| **Fig 1** | Architecture Diagram | Your system architecture (already in README) |
| **Fig 2** | Bar Chart | Retrieval Recall@5 across 5 embedding models |
| **Fig 3** | Bar Chart | Answer quality (Faithfulness, Relevancy) across 5 LLMs |
| **Fig 4** | Scatter Plot | UMAP projection of embeddings colored by HDBSCAN vs K-Means legal topic clusters |
| **Fig 5** | Heatmap | 3×3 LLM × Embedder full-pipeline accuracy |
| **Fig 6** | Grouped Bar | Component ablation — metric drops when each module is disabled |
| **Fig 7** | Box Plot | Latency distribution per pipeline stage |
| **Fig 8** | Spider/Radar Chart | Multi-metric comparison of top 3 configurations |
| **Table 1** | Summary Table | All models, all metrics, single comparison table |

### Phase 5: Paper Writing (Estimated: 3-4 days)

---

## 6. Suggested Paper Structure

```
Title: Privacy-First RAG for Indian Legal Aid: A Conflict-Aware Pipeline
       with Local Models for Women and Child Law

Abstract (250 words)

1. Introduction
   - Legal AI gap in Indian context
   - Privacy concerns with cloud-based legal AI
   - Contributions (N1–N5)

2. Related Work
   - Legal RAG systems (cite your 5 benchDocs papers)
   - Privacy in NLP pipelines
   - Indian legal NLP

3. System Architecture
   - 3.1 Ingestion Pipeline (hierarchical parsing, metadata enrichment)
     - 3.2 Query Pipeline (rewriting → retrieval → conflicts → generation)

4. Experimental Setup
   - 4.1 Dataset Construction (100 QA pairs, annotation protocol)
     - 4.2 Models Under Study (5 LLMs, 5 embedders, HDBSCAN vs KMeans)

  5. Results
     - 5.1 Retrieval Performance (Fig: Recall@k per embedder)
     - 5.2 Generation Quality (Fig: RAGAS metrics per LLM)
      - 5.3 Semantic Clustering (Table: HDBSCAN vs K-means purity, noise ratio)
      - 5.4 Cross-Ablation (Fig: LLM × Embedder heatmap)
      - 5.5 Component Ablation (Fig: what each module adds)
   - 5.6 Privacy Guarantees (zero leakage rate)
   - 5.7 Latency Analysis

6. Discussion
   - Key findings
   - Retrieval > Generation (cite Legal RAG Bench)
   - Privacy-accuracy tradeoff analysis

7. Limitations
   - Single domain (Indian family/child law)
   - Self-annotated dataset (no legal expert validation)
   - Local LLM as RAGAS judge (not GPT-4)
   - Small corpus (8 documents)

8. Conclusion & Future Work

References (~30-40 papers)
```

---

## 7. Timeline Summary

| Phase | Activity | Duration | Human Effort |
|-------|---------|----------|-------------|
| **P1** | Gold dataset creation (read PDFs, write 100 QA pairs) | 2-3 days | High (manual annotation) |
| **P2** | Benchmark harness scripts | 3-4 days | Medium (coding) |
| **P3** | Run experiments | 3-5 days | Low (mostly waiting for models) |
| **P4** | Analysis & visualization | 2 days | Medium (analysis) |
| **P5** | Paper writing | 3-4 days | High (writing) |
| **Total** | | **~15-18 days** | |

---

## 8. Key References to Cite (From Your benchDocs)

| Paper | ArXiv ID | Key Citation Point |
|-------|----------|-------------------|
| **LegalBench-RAG** | 2408.10343 | First legal retrieval benchmark; character-level precision/recall; cite for methodology inspiration |
| **LegalRAG (Bangla)** | 2504.16121 | Multilingual legal RAG; advanced vs vanilla RAG comparison; cite for multilingual parallels |
| **Revisiting RAG Retrievers** | 2602.23371 | Mutual-information retriever analysis; cite for retriever ensemble theory |
| **Legal RAG Bench** | 2603.01710 | Retrieval as primary driver of E2E performance; hierarchical error decomposition; **cite heavily** — closest to your work |
| **Benchmarking Legal RAG (Statutory Surveys)** | 2603.03300 | STARA system; ground truth quality issues; cite for evaluation methodology lessons |

---

## Open Questions for You

> [!WARNING]
> **Q1**: What is your GPU hardware? This affects which of the 14B models you can run. If CPU-only, stick to 7B-8B models and drop `phi4:14b` and `gemma3:12b`.

> [!WARNING]  
> **Q2**: Do you want to also test **hybrid retrieval** (semantic + BM25 keyword search) as an additional ablation dimension? This is a common ask from reviewers and would add ~1 day of work.

> [!IMPORTANT]
> **Q3**: For the gold dataset — are you comfortable reading through the 8 PDFs and annotating 100 QA pairs yourself? This requires ~6-8 hours of focused reading. Alternatively, we can start with 50 and expand later.

> [!NOTE]
> **Q4**: Which target venue/conference are you considering? This affects the page limit, formatting, and emphasis (e.g., ACL workshops want more NLP depth; CIKM wants more systems depth; a journal like JAIR allows longer papers).
