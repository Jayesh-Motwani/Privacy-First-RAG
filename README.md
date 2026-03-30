# Privacy-First Legal RAG Pipeline
## Indian Family & Child Law Assistant

A complete **Retrieval-Augmented Generation (RAG)** pipeline for Indian legal documents using **100% free, open-source models** with privacy-first design. Built specifically for Indian women and child law, covering Hindu Marriage Act, Domestic Violence Act, CrPC, POCSO, Juvenile Justice Act, and related legislation.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Module Breakdown](#module-breakdown)
- [Data Flow](#data-flow)
- [Installation](#installation)
- [Usage](#usage)
- [PII Model Integration](#pii-model-integration)
- [Conflict Detection System](#conflict-detection-system)
- [Sample Documents](#sample-documents)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

### Privacy-First Design
- **Local PII Masking**: Uses fine-tuned BERT NER model (`pii_bert_model/`) to detect and mask 112+ entity types before query processing
- **No Cloud APIs**: All models run locally on your machine - no data leaves your system
- **Local Embeddings**: HuggingFace sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) for document embeddings
- **Local LLM**: Ollama with qwen2.5:7b or mistral:7b for query rewriting and answer generation

### Legal Document Processing
- **PDF Ingestion**: Automatic text extraction from PDF legal documents using PyMuPDF
- **Hierarchical Parsing**: Intelligent parsing of Act → Part → Chapter → Section → Sub-section → Proviso structure
- **Smart Metadata Enrichment**: Auto-classifies provisions by:
  - **Personal Law**: Hindu, Muslim, Christian, Secular, Special Marriage Act
  - **Demographic Scope**: Married women, minors, divorced women, working women
  - **Law Type**: Civil, Criminal, Procedural
  - **Jurisdiction**: Central vs State legislation

### Advanced RAG Features
- **Conflict Detection**: Identifies tensions between retrieved legal provisions (e.g., HMA vs DV Act, POCSO vs CrPC)
- **Applicability Filtering**: Filters retrieval results based on user's jurisdiction, personal law, and demographic profile
- **Legal Query Rewriting**: Converts layperson queries into precise legal terminology using LLM
- **Multi-Stage Retrieval**: Combines semantic similarity with metadata filtering for accurate results

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PRIVACY-FIRST LEGAL RAG PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  INGESTION PIPELINE (Run Once)                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐   │
│  │  PDF Files   │───▶│  PDF Extractor   │───▶│  HierarchicalDocumentParser │   │
│  │  (data/)     │    │  (PyMuPDF/pypdf) │    │  (Act>Part>Chapter>Section) │   │
│  └──────────────┘    └──────────────────┘    └─────────────────────────────┘   │
│                                                        │                        │
│                                                        ▼                        │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────────┐   │
│  │   ChromaDB       │◀───│  Embedding Model │◀───│  MetadataEnricher       │   │
│  │   (Vector Store) │    │  (sentence-trans)│    │  (Auto-classification)  │   │
│  └──────────────────┘    └──────────────────┘    └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  QUERY PIPELINE (Per Query)                                                     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐   │
│  │  User Query  │───▶│  PII Masker      │───▶│  Legal Query Rewriter       │   │
│  │  + Profile   │    │  (BERT NER +     │    │  (Ollama LLM)               │   │
│  │              │    │   Regex Fallback)│    │                             │   │
│  └──────────────┘    └──────────────────┘    └─────────────────────────────┘   │
│                                                        │                        │
│                                                        ▼                        │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────────┐   │
│  │  Final Answer    │◀───│  LLM Generator   │◀───│  Conflict Detector      │   │
│  │  (with citations)│    │  (Ollama qwen2.5)│    │  (25 known conflicts)   │   │
│  └──────────────────┘    └──────────────────┘    └─────────────────────────┘   │
│                                ▲                                                │
│                                │                                                │
│  ┌─────────────────────────────┴────────────────────────────────────────────┐   │
│  │                    Semantic Retrieval + Metadata Filter                   │   │
│  │                    (ChromaDB with applicability filtering)                │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **LLM** | Ollama (qwen2.5:7b / mistral:7b) | Query rewriting, answer generation |
| **Embeddings** | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Semantic search |
| **Vector Store** | ChromaDB | Document storage with metadata filtering |
| **NER Model** | Fine-tuned BERT (pii_bert_model/) | PII detection and masking |
| **PDF Parser** | PyMuPDF + pypdf | Text extraction from legal PDFs |
| **Orchestration** | LangChain | RAG pipeline coordination |

---

## Module Breakdown

### 1. `pdf_extractor.py` - PDF Text Extraction

**Purpose**: Extracts text from PDF legal documents with special handling for Indian court judgments, legislative acts, and government notifications.

**Key Classes**:
- `PDFTextExtractor`: Main extraction class with PyMuPDF/pypdf backends
- `ActNameExtractor`: Detects act names from document content
- `ExtractedDocument`: Dataclass for extracted documents with metadata

**Features**:
- Multi-column layout handling
- Page header/footer removal
- Document type classification (Act, Judgment, Notification)
- Metadata extraction (act name, year, document type)

```python
# Usage
from pdf_extractor import PDFTextExtractor

extractor = PDFTextExtractor(data_dir="./data")
documents = extractor.extract_all()  # Returns list[ExtractedDocument]
```

---

### 2. `ingest.py` - Document Ingestion Pipeline

**Purpose**: Parses legal documents into hierarchical chunks, enriches with metadata, generates embeddings, and stores in ChromaDB.

**Key Classes**:

#### `HierarchicalDocumentParser`
Parses Indian legal documents into hierarchical structure:
- **Patterns detected**: Part, Chapter, Section, Sub-section, Clause, Proviso, Explanation, Illustration
- **Output**: List of `Document` objects with hierarchical metadata

```python
# Internal hierarchy detection patterns
PATTERNS = {
    'part': re.compile(r'^PART\s+(?:[IVX]+|[0-9]+|[-\s]*)[\s\.-]+(.+)$'),
    'chapter': re.compile(r'^CHAPTER\s+(?:[IVX]+|[0-9]+|[-\s]*)[\s\.-]+(.+)$'),
    'section_main': re.compile(r'^(?:Section|Sec\.?|S\.?)?\s*([0-9]+(?:[A-Z])?)\.?\s*[-\s]*(.*)$'),
    'subsection': re.compile(r'^\(([0-9]+|[a-z]|[A-Z]|[ivx]+)\)\s*(.*)$'),
    'proviso': re.compile(r'^(?:Provided\s+that|Provided\s+further\s+that)\s*(.*)$'),
}
```

#### `MetadataEnricher`
Auto-classifies legal provisions by:
- **Personal Law**: Detects Hindu, Muslim, Christian, Secular applicability
- **Demographic Scope**: Identifies provisions for married women, minors, etc.
- **Law Type**: Classifies as civil, criminal, or procedural

#### `LegalChunkMetadata` (Dataclass)
Schema for legal document chunk metadata:
```python
@dataclass
class LegalChunkMetadata:
    act_name: str
    part: Optional[str]
    chapter: Optional[str]
    section_number: str
    section_title: str
    subsection: Optional[str]
    jurisdiction: str  # "central" or "state:<name>"
    applicable_personal_law: list
    demographic_scope: list
    law_type: str  # "civil" | "criminal" | "procedural"
```

**Pipeline Flow**:
```
PDF Text → Hierarchical Parsing → Metadata Enrichment → Embedding → ChromaDB
```

---

### 3. `query_pipeline.py` - Query Processing Pipeline

**Purpose**: Processes user queries through PII masking, query rewriting, semantic retrieval, conflict detection, and answer generation.

**Key Classes**:

#### `PIIMasker`
Privacy-first PII masking using fine-tuned BERT NER model.

**Features**:
- Loads model from `pii_bert_model/checkpoint-26160`
- Supports 112 entity types (FIRSTNAME, LASTNAME, DOB, PHONE, etc.)
- Regex fallbacks for Indian IDs (Aadhaar, PAN, Voter ID, Passport)
- Entity mapping to legal-friendly placeholders

```python
# Entity mapping examples
LABEL_MAP = {
    'FIRSTNAME': '[PERSON]',
    'PHONENUMBER': '[PHONE]',
    'AADHAAR': '[ID]',
    'PAN': '[ID]',
}
```

#### `LegalQueryRewriter`
Converts layperson queries into precise legal terminology.

**Prompt Template**:
```python
REWRITE_PROMPT = PromptTemplate(
    input_variables=["masked_query"],
    template="""You are a legal terminology assistant for Indian women and child law.
Your task is to rewrite layperson queries into precise legal terminology.

Guidelines:
1. Identify the legal issue(s) in the query
2. Reference relevant Indian acts (DV Act, HMA, CrPC, POCSO, JJ Act, etc.)
3. Use proper legal terminology
4. Keep the query concise but legally precise

Query: {masked_query}

Rewritten query:"""
)
```

#### `LegalQueryPipeline`
Main orchestrator class that coordinates the full query pipeline.

**Methods**:
- `process_query(query, user_profile)`: Main entry point
- `_build_filter_criteria(profile)`: Builds ChromaDB metadata filters
- `_detect_conflicts(chunks)`: Checks retrieved chunks against conflict map
- `_generate_answer(query, chunks, conflicts)`: Generates final answer with RAG

---

### 4. `run_pipeline.py` - Main Entry Point

**Purpose**: CLI interface for running the full pipeline, ingestion-only, query-only, or interactive modes.

**Modes**:
```bash
# Full pipeline (ingest + demo queries)
python run_pipeline.py

# Ingestion only
python run_pipeline.py --ingest-only --clear

# Query demo only (use existing database)
python run_pipeline.py --query-only

# Interactive mode
python run_pipeline.py --interactive

# List documents
python run_pipeline.py --list-docs
```

---

### 5. `conflict_map.json` - Conflict Detection Database

**Purpose**: Pre-defined map of known conflicts between legal provisions.

**Structure**:
```json
[
  {
    "provision_a": "HMA_Section_13",
    "provision_b": "DV_Act_Section_17",
    "note": "Divorce under HMA Section 13 may conflict with right to residence under DV Act Section 17"
  },
  {
    "provision_a": "CrPC_Section_125",
    "provision_b": "DV_Act_Section_20",
    "note": "Maintenance under CrPC Section 125 and DV Act Section 20 can be claimed concurrently"
  }
]
```

**Coverage**: 25 known conflicts covering:
- HMA vs DV Act (divorce, maintenance, residence rights)
- CrPC vs DV Act (maintenance overlap)
- POCSO vs CrPC (bail, reporting, compensation)
- HAMA vs JJ Act (adoption frameworks)
- Hindu Succession vs Indian Succession (property rights)

---

## Data Flow

### Ingestion Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PDF EXTRACTION                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: ./data/*.pdf                                                         │
│ Process: PyMuPDF extracts raw text + metadata                               │
│ Output: List[ExtractedDocument]                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: HIERARCHICAL PARSING                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: Raw document text                                                    │
│ Process: HierarchicalDocumentParser detects Act>Part>Chapter>Section        │
│ Output: List[Document] with hierarchical structure                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: METADATA ENRICHMENT                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: Parsed documents                                                     │
│ Process: MetadataEnricher auto-classifies by personal law, demographic      │
│ Output: Documents with enriched metadata                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: EMBEDDING GENERATION                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: Document chunks                                                      │
│ Process: sentence-transformers generates 384-dim embeddings                 │
│ Output: Embeddings + metadata                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: VECTOR STORAGE                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: Embeddings + metadata                                                │
│ Process: ChromaDB stores with metadata indexing                             │
│ Output: Persistent vector database (./chroma_db)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Query Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PII MASKING                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: "My husband Ramesh beats me. Call me at 9876543210"                  │
│ Process: BERT NER + regex detects names, phones, IDs                        │
│ Output: "My husband [PERSON] beats me. Call me at [PHONE]"                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: QUERY REWRITING                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: Masked query + user profile                                          │
│ Process: Ollama LLM rewrites to legal terminology                           │
│ Output: "What remedies are available under DV Act Section 23 for physical   │
│         abuse and what protection orders can be sought under Section 18?"   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: APPLICABILITY FILTERING                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: User profile (jurisdiction, personal law, demographic)               │
│ Process: Build ChromaDB where-clause from profile                           │
│ Output: Filter criteria {"personal_law": ["hindu"], "demographic": [...]}   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: SEMANTIC RETRIEVAL                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: Rewritten query + filter criteria                                    │
│ Process: ChromaDB similarity search with metadata filtering                 │
│ Output: Top-k relevant legal provisions (chunks)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: CONFLICT DETECTION                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: Retrieved chunks                                                     │
│ Process: Check chunk pairs against conflict_map.json                        │
│ Output: List of detected conflicts with notes                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: ANSWER GENERATION                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: Original query, retrieved chunks, conflicts                          │
│ Process: Ollama LLM generates answer with RAG context                       │
│ Output: Final answer with legal citations and conflict warnings             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies**:
- `langchain>=0.2.0` - RAG orchestration
- `langchain-huggingface>=0.0.3` - HuggingFace embeddings
- `langchain-ollama>=0.1.0` - Ollama LLM integration
- `langchain-chroma` - ChromaDB vector store
- `chromadb>=0.4.0` - Vector database
- `transformers>=4.35.0` - BERT models
- `sentence-transformers>=2.2.0` - Embedding models
- `torch>=2.0.0` - PyTorch backend
- `pymupdf>=1.23.0` - PDF extraction
- `pypdf>=3.0.0` - PDF fallback parser

### 2. Install Ollama

**Windows**:
1. Download from https://ollama.ai
2. Run the installer
3. Verify installation:
```bash
ollama list
```

**Pull Required Model**:
```bash
ollama pull qwen2.5:7b
# Alternative:
ollama pull mistral:7b
```

### 3. Verify Setup

```bash
# List PDFs in data folder
python run_pipeline.py --list-docs

# Test PII model
python test_pii_model.py
```

---

## Usage

### Full Pipeline (Ingest + Query Demo)

```bash
python run_pipeline.py
```

This runs:
1. PDF ingestion from `./data` folder
2. Embedding generation and ChromaDB storage
3. Demo query processing with sample questions

### Ingest PDFs Only

```bash
# Ingest all PDFs from data/ folder
python run_pipeline.py --ingest-only

# Re-ingest (clear old database first)
python run_pipeline.py --ingest-only --clear
```

### Query Only (Use Existing Database)

```bash
python run_pipeline.py --query-only
```

### Interactive Mode

```bash
python run_pipeline.py --interactive
```

Enter your user profile once, then ask multiple questions:
```
Set up your user profile (press Enter for defaults):
  Jurisdiction [central]: Karnataka
  Personal Law [all]: hindu
  Demographic [any]: married_woman

Your question: My husband doesn't give me money for food. What can I do?
```

### Programmatic Usage

```python
from query_pipeline import LegalQueryPipeline

# Initialize pipeline
pipeline = LegalQueryPipeline()

# Process a query
result = pipeline.process_query(
    user_query="My husband beats me. What are my rights?",
    user_profile={
        "jurisdiction": "Karnataka",
        "personal_law": "hindu",
        "demographic": "married_woman"
    }
)

# Access results
print(result['answer'])           # Generated answer
print(result['rewritten_query'])  # Legal terminology version
print(result['conflicts'])        # Detected provision conflicts
print(result['retrieved_chunks']) # Retrieved legal provisions
print(result['pii_masked'])       # PII-masked query
print(result['entities_found'])   # PII entities detected
```

---

## PII Model Integration

### Model Architecture

The pipeline uses your fine-tuned PII BERT model from `pii_bert_model/checkpoint-26160/`.

**Model Details**:
- **Base**: BERT-based NER model
- **Training**: Fine-tuned on PII entity recognition
- **Entities**: 112 entity types
- **Location**: Local folder (no HuggingFace download required)

### Supported Entity Types

| Category | Entities |
|----------|----------|
| **Personal** | FIRSTNAME, LASTNAME, DOB, AGE, GENDER |
| **Location** | CITY, STATE, STREET, ZIPCODE, BUILDINGNUMBER |
| **Contact** | PHONENUMBER, EMAIL, URL, USERAGENT |
| **Financial** | ACCOUNTNUMBER, CREDITCARDNUMBER, IBAN, CURRENCY |
| **Identifiers** | SSN, PASSWORD, PIN, IP, MAC |
| **Indian IDs** (regex) | Aadhaar, PAN, Voter ID, Passport |

### Entity Mapping

Model labels are mapped to legal-friendly placeholders:

```python
LABEL_MAP = {
    'FIRSTNAME': '[PERSON]',
    'LASTNAME': '[PERSON]',
    'CITY': '[LOCATION]',
    'STATE': '[LOCATION]',
    'PHONENUMBER': '[PHONE]',
    'AADHAAR': '[ID]',
    'PAN': '[ID]',
    'ACCOUNTNUMBER': '[ACCOUNT]',
    'CREDITCARDNUMBER': '[CARD]',
}
```

### Testing the PII Model

```bash
python test_pii_model.py
```

Example output:
```
Input: "My name is Rahul Sharma and my Aadhaar is 1234 5678 9012"
Masked: "My name is [PERSON] [PERSON] and my Aadhaar is [ID]"
Entities: {'FIRSTNAME': 1, 'LASTNAME': 1, 'aadhaar': 1}
```

---

## Conflict Detection System

### Overview

The `conflict_map.json` contains **25 known conflicts** in Indian family and child law. The system automatically detects when retrieved provisions have known tensions and includes warnings in the generated answer.

### Conflict Categories

| Category | Conflicts |
|----------|-----------|
| **HMA vs DV Act** | Divorce vs residence, restitution vs protection |
| **CrPC vs DV Act** | Maintenance overlap, enforcement procedures |
| **POCSO vs CrPC** | Bail stringency, reporting requirements, compensation |
| **HAMA vs JJ Act** | Adoption frameworks (religious vs secular) |
| **Succession Laws** | Hindu vs Christian property rights |
| **Muslim Law vs CrPC** | Maintenance after divorce (Danial Latifi ruling) |

### Example Conflicts

```json
{
  "provision_a": "HMA_Section_9",
  "provision_b": "DV_Act_Section_18",
  "note": "Restitution of conjugal rights under HMA Section 9 may conflict with protection orders under DV Act Section 18 - woman's safety and protection from domestic violence takes precedence over conjugal rights"
}
```

### Runtime Detection

During query processing:
1. Retrieved chunks are checked against `conflict_map.json`
2. If both provisions of a conflict pair are retrieved, the conflict is flagged
3. The LLM is instructed to mention the conflict in the generated answer

---

## Sample Documents

The `data/` folder includes:

| Document | Description |
|----------|-------------|
| `protection_of_women_from_domestic_violence_act,_2005.pdf` | DV Act - Protection orders, residence rights, monetary relief |
| `the_code_of_criminal_procedure,_1973.pdf` | CrPC - FIR, maintenance (Sec 125), bail procedures |
| `A1955-25.pdf` | Hindu Marriage Act, 1955 - Marriage, divorce, maintenance |
| `AA2012-32.pdf` | Hindu Adoptions and Maintenance Act - Adoption, maintenance |
| `A2013-14.pdf` | Related family law legislation |
| `a2016-2.pdf` | Recent amendments and acts |
| `showfile(1).pdf` | Court judgments and case law |

---

## Troubleshooting

### Ollama Not Found
```bash
# Install Ollama
# Windows: Download from https://ollama.ai
# Then verify installation
ollama list

# If not found, add to PATH or restart terminal
```

### Model Not Available
```bash
# Pull required model
ollama pull qwen2.5:7b

# Or use alternative
ollama pull mistral:7b
```

### No PDFs Found
```bash
# Check data directory
python run_pipeline.py --list-docs

# Specify custom data directory
python run_pipeline.py --data /path/to/pdfs
```

### PII Model Loading Error
```bash
# Test PII model
python test_pii_model.py

# Check model path in query_pipeline.py
# Default: ./pii_bert_model/checkpoint-26160
```

### ChromaDB Errors
```bash
# Clear and re-ingest
python run_pipeline.py --ingest-only --clear

# Or manually delete chroma_db folder
rm -rf ./chroma_db  # Linux/Mac
rmdir /s ./chroma_db  # Windows
```

### PyMuPDF Not Installed
```bash
pip install pymupdf
```

### Embedding Model Download Fails
```bash
# The model downloads on first run. If it fails:
# 1. Check internet connection
# 2. Clear HuggingFace cache
# 3. Re-run ingestion

# Clear HF cache (optional)
rm -rf ~/.cache/huggingface/hub  # Linux/Mac
```

---

## Project Structure

```
Privacy-First-RAG/
├── data/                           # PDF legal documents
│   ├── 250883_english_01042024.pdf
│   ├── A1955-25.pdf               # Hindu Marriage Act, 1955
│   ├── A2013-14.pdf
│   ├── a2016-2.pdf
│   ├── AA2012-32.pdf              # Hindu Adoptions & Maintenance Act
│   ├── protection_of_women_from_domestic_violence_act,_2005.pdf
│   ├── showfile(1).pdf
│   └── the_code_of_criminal_procedure,_1973.pdf
│
├── pii_bert_model/                 # Fine-tuned PII NER model
│   └── checkpoint-26160/           # Model checkpoint
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
│
├── chroma_db/                      # Vector database (created after ingestion)
│   ├── chroma.sqlite3
│   └── ...
│
├── final_model/                    # Additional model artifacts
│
├── conflict_map.json               # 25 known provision conflicts
├── requirements.txt                # Python dependencies
│
├── pdf_extractor.py                # PDF text extraction module
├── ingest.py                       # Document ingestion pipeline
├── query_pipeline.py               # Query processing pipeline
├── run_pipeline.py                 # Main CLI entry point
├── test_pii_model.py               # PII model test script
└── README.md                       # This documentation
```

---

## License

This project uses only open-source components:
- **Models**: Apache 2.0 / MIT licensed (HuggingFace, Ollama)
- **Libraries**: Standard open-source licenses (LangChain, ChromaDB, PyMuPDF)
- **Legal Documents**: Public domain (Indian legislation)

Your fine-tuned models and added legal documents remain your property.

---

## Contributing

### Adding Legal Documents
1. Place PDF files in `data/` folder
2. Run: `python run_pipeline.py --ingest-only --clear`

### Updating Conflict Map
1. Edit `conflict_map.json` with new provision pairs
2. Include a descriptive note explaining the conflict
3. Use format: `{"provision_a": "...", "provision_b": "...", "note": "..."}`

### Using Different PII Model
1. Update `PII_MODEL_PATH` in `query_pipeline.py`
2. Or pass to constructor: `PIIMasker(model_path="./pii_bert_model/other-checkpoint")`

### Changing Embedding Model
1. Update `EMBEDDING_MODEL_NAME` in `ingest.py` and `query_pipeline.py`
2. Recommended: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

---

## Performance Notes

| Operation | Time (approx.) |
|-----------|----------------|
| PDF Ingestion (8 docs) | 2-5 minutes |
| Embedding Generation | 30-60 seconds |
| Query Processing | 5-15 seconds |
| PII Masking | <1 second |
| Query Rewriting | 2-5 seconds |
| Retrieval | <1 second |
| Answer Generation | 3-10 seconds |

**Hardware Requirements**:
- **Minimum**: 8GB RAM, CPU only
- **Recommended**: 16GB RAM, GPU for faster LLM inference
- **Storage**: ~2GB for models + database

---

## Research & Novelty

This pipeline implements several novel features for Indian legal tech:

1. **Privacy-First PII Masking**: First Indian legal RAG to use fine-tuned BERT NER for PII detection before query processing
2. **Hierarchical Legal Parsing**: Custom parser for Indian legislative structure (Act>Part>Chapter>Section>Proviso)
3. **Conflict-Aware RAG**: Automatic detection of tensions between legal provisions during retrieval
4. **Applicability Filtering**: Metadata-based filtering by personal law, jurisdiction, and demographic
5. **100% Local Processing**: No cloud APIs - all models run on user's machine
