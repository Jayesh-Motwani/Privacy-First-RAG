import re
with open('ingest.py', 'r') as f:
    text = f.read()

text = text.replace(
    'def run_ingestion(data_dir: str = "./data", clear_first: bool = False):',
    'def run_ingestion(data_dir: str = "./data", clear_first: bool = False, persist_directory: str = "./chroma_db", embedding_model: str = None):'
)

text = text.replace(
    'ingestor = LegalDocumentIngestor(data_dir=data_dir)',
    'ingestor = LegalDocumentIngestor(data_dir=data_dir, persist_directory=persist_directory, embedding_model=embedding_model)'
)

with open('ingest.py', 'w') as f:
    f.write(text)
