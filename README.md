# Privacy-First Legal RAG Pipeline
## Indian Family & Child Law Assistant

A complete RAG (Retrieval-Augmented Generation) pipeline for Indian legal documents using **100% free, open-source models** with privacy-first design.

---

## Features

### Privacy-First Design
- **Local PII Masking**: Uses your fine-tuned BERT model (`pii_bert_model/`) to detect and mask personal information before query processing
- **No Cloud APIs**: All models run locally - no data leaves your machine
- **Local Embeddings**: HuggingFace sentence-transformers for document embeddings
- **Local LLM**: Ollama with qwen2.5:7b or mistral:7b for query rewriting and generation

### Legal Document Processing
- **PDF Ingestion**: Automatically extracts text from PDF legal documents
- **Hierarchical Parsing**: Parses Act → Part → Chapter → Section → Sub-section structure
- **Smart Metadata**: Auto-classifies provisions by:
  - Personal law (Hindu, Muslim, Christian, Secular)
  - Demographic scope (married women, minors, working women, etc.)
  - Law type (civil, criminal, procedural)

### Advanced Features
- **Conflict Detection**: Identifies tensions between legal provisions (e.g., HMA vs DV Act)
- **Applicability Filtering**: Filters results based on user's jurisdiction, personal law, and demographic
- **Legal Query Rewriting**: Converts layperson queries into precise legal terminology

---

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

Download from: https://ollama.ai

Then pull a model:
```bash
ollama pull qwen2.5:7b
# or
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

### Programmatic Usage

```python
from query_pipeline import process_query

# Process a query
result = process_query(
    user_query="My husband beats me. What are my rights?",
    user_profile={
        "jurisdiction": "Karnataka",
        "personal_law": "hindu",
        "demographic": "married_woman"
    }
)

print(result['answer'])
print(result['rewritten_query'])  # Legal terminology version
print(result['conflicts'])  # Detected provision conflicts
```

---

## Project Structure

```
Privacy-First-RAG/
├── data/                       # PDF legal documents
│   ├── protection_of_women_from_domestic_violence_act,_2005.pdf
│   ├── the_code_of_criminal_procedure,_1973.pdf
│   └── ...
├── pii_bert_model/             # Your fine-tuned PII model
│   └── checkpoint-26160/       # Model checkpoint
├── chroma_db/                  # Vector database (created after ingestion)
├── conflict_map.json           # Known provision conflicts
├── ingest.py                   # Document ingestion pipeline
├── query_pipeline.py           # Query processing pipeline
├── pdf_extractor.py            # PDF text extraction
├── run_pipeline.py             # Main entry point
├── test_pii_model.py           # PII model test script
└── requirements.txt            # Dependencies
```

---

## Pipeline Architecture

### Ingestion Pipeline (Run Once)

1. **PDF Extraction**: Extract text from PDFs using PyMuPDF
2. **Act Name Detection**: Automatically identify act names from document content
3. **Hierarchical Parsing**: Parse document structure (Act > Part > Chapter > Section)
4. **Metadata Enrichment**: Auto-classify provisions by personal law, demographic, law type
5. **Embedding Generation**: Create embeddings using sentence-transformers
6. **Vector Storage**: Store in ChromaDB with metadata

### Query Pipeline (Per Query)

1. **PII Masking**: Detect and mask personal info using your fine-tuned BERT model
2. **Query Rewriting**: Convert to legal terminology using Ollama LLM
3. **Applicability Filtering**: Build ChromaDB filter from user profile
4. **Semantic Retrieval**: Retrieve top-k relevant provisions
5. **Conflict Detection**: Check for tensions between retrieved provisions
6. **LLM Generation**: Generate answer with RAG context

---

## PII Model Integration

The pipeline uses your fine-tuned PII BERT model from `pii_bert_model/checkpoint-26160/`.

### Supported Entity Types

Your model detects 112 entity types including:
- **Personal**: FIRSTNAME, LASTNAME, DOB, AGE, GENDER
- **Location**: CITY, STATE, STREET, ZIPCODE
- **Contact**: PHONENUMBER, EMAIL, URL
- **Financial**: ACCOUNTNUMBER, CREDITCARDNUMBER, CURRENCY
- **Identifiers**: SSN, PASSWORD, PIN, IP, MAC
- **Indian IDs** (regex fallback): Aadhaar, PAN, Voter ID, Passport

### Entity Mapping

Model labels are mapped to legal-friendly placeholders:
- `FIRSTNAME`, `LASTNAME` → `[PERSON]`
- `CITY`, `STATE`, `STREET` → `[LOCATION]`
- `PHONENUMBER`, `PHONEIMEI` → `[PHONE]`
- `ACCOUNTNUMBER`, `IBAN` → `[ACCOUNT]`

---

## Conflict Map

The `conflict_map.json` contains 25 known conflicts in Indian law:

| Provision A | Provision B | Conflict Summary |
|-------------|-------------|------------------|
| HMA_Section_13 | DV_Act_Section_17 | Divorce vs residence rights |
| CrPC_Section_125 | DV_Act_Section_20 | Maintenance overlap |
| HMA_Section_9 | DV_Act_Section_18 | Restitution vs protection |
| POCSO_Section_29 | CrPC_Section_167 | Bail stringency |
| HAMA_Section_6 | JJ_Act_Section_56 | Adoption frameworks |

---

## Sample Documents

The `data/` folder includes:
- Protection of Women from Domestic Violence Act, 2005
- Code of Criminal Procedure, 1973
- Hindu Marriage Act, 1955
- Hindu Adoptions and Maintenance Act, 1956
- Juvenile Justice Act, 2015
- Other Indian legal acts

---

## Troubleshooting

### Ollama Not Found
```bash
# Install Ollama
# Windows: Download from https://ollama.ai
# Then verify installation
ollama list
```

### Model Not Available
```bash
ollama pull qwen2.5:7b
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
```

---

## Dependencies

All dependencies are free and open-source:

| Package | Purpose |
|---------|---------|
| langchain | RAG orchestration |
| langchain-huggingface | HF embeddings |
| langchain-ollama | Local LLM |
| chromadb | Vector database |
| transformers | BERT models |
| sentence-transformers | Embeddings |
| pymupdf | PDF extraction |
| pypdf | PDF fallback |

---

## License

This project uses only open-source components. Your fine-tuned models and legal documents remain your property.

---

## Contributing

To add more legal documents:
1. Place PDF files in `data/` folder
2. Run: `python run_pipeline.py --ingest-only --clear`

To update conflict map:
1. Edit `conflict_map.json` with new provision pairs
2. Include a descriptive note explaining the conflict

To use a different PII model checkpoint:
1. Update `PII_MODEL_PATH` in `query_pipeline.py`
2. Or pass to constructor: `PIIMasker(model_path="./pii_bert_model/other-checkpoint")`
