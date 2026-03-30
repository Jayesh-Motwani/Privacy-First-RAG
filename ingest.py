"""
Legal Document Ingestion Pipeline
=================================
Parses Indian legal documents from PDF files into hierarchical chunks,
embeds them using free HuggingFace models, and stores in ChromaDB with
rich metadata.

Novelty modules:
1. HierarchicalDocumentParser - Parses Act > Part > Chapter > Section > Sub-section > Proviso
2. MetadataEnricher - Adds jurisdiction, personal law, demographic scope metadata
3. ConflictMapLoader - Loads known provision conflicts for runtime detection
4. PDFActExtractor - Extracts act names and metadata from PDFs
"""

import json
import re
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb

# Local imports
from pdf_extractor import PDFTextExtractor, ActNameExtractor, ExtractedDocument


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LegalChunkMetadata:
    """Metadata schema for legal document chunks."""
    act_name: str
    part: Optional[str] = None
    chapter: Optional[str] = None
    section_number: str = ""
    section_title: str = ""
    subsection: Optional[str] = None
    chunk_text: str = ""
    jurisdiction: str = "central"  # "central" | "state:<name>"
    applicable_personal_law: list = field(default_factory=list)
    demographic_scope: list = field(default_factory=list)
    law_type: str = "civil"  # "civil" | "criminal" | "procedural"

    def to_dict(self) -> dict:
        """Convert to dictionary for ChromaDB storage."""
        return asdict(self)


# =============================================================================
# HIERARCHICAL DOCUMENT PARSER
# =============================================================================

class HierarchicalDocumentParser:
    """
    Parses Indian legal documents into hierarchical structure.
    
    Hierarchy levels detected:
    - Act (top level)
    - Part (e.g., "PART I - PRELIMINARY")
    - Chapter (e.g., "CHAPTER II - MARRIAGE")
    - Section (e.g., "Section 1. Short title")
    - Sub-section (e.g., "(1)", "(2)")
    - Proviso (e.g., "Provided that")
    """
    
    # Regex patterns for hierarchy detection - optimized for Indian legal PDFs
    PATTERNS = {
        # Part: "PART I", "PART - I", "PART 1", "PART - PRELIMINARY"
        'part': re.compile(r'^PART\s+(?:[IVX]+|[0-9]+|[-\s]*)[\s\.-]+(.+)$', re.IGNORECASE | re.MULTILINE),
        
        # Chapter: "CHAPTER II", "CHAPTER - II", "CHAPTER 1"
        'chapter': re.compile(r'^CHAPTER\s+(?:[IVX]+|[0-9]+|[-\s]*)[\s\.-]+(.+)$', re.IGNORECASE | re.MULTILINE),
        
        # Main section: "Section 1.", "Sec. 1", "S. 1", "1." at line start
        'section_main': re.compile(r'^(?:Section|Sec\.?|S\.?|§)?\s*([0-9]+(?:[A-Z])?)\.?\s*[-\s]*(.*)$', re.IGNORECASE | re.MULTILINE),
        
        # Numbered line that might be a section
        'section_numbered': re.compile(r'^([0-9]+)\.?\s+[-\s]*(.+)$', re.MULTILINE),
        
        # Subsection: "(1)", "(a)", "(i)"
        'subsection': re.compile(r'^\(([0-9]+|[a-z]|[A-Z]|[ivx]+)\)\s*(.*)$', re.MULTILINE),
        
        # Clause: "(i)", "(ii)", "(a)", "(b)"
        'clause': re.compile(r'^\(([ivx]+|[0-9]+|[a-z]|[A-Z])\)\s*(.*)$', re.IGNORECASE | re.MULTILINE),
        
        # Proviso
        'proviso': re.compile(r'^(?:Provided\s+that|Provided\s+further\s+that)\s*(.*)$', re.IGNORECASE | re.MULTILINE),
        
        # Explanation
        'explanation': re.compile(r'^(?:Explanation\s*[0-9]*|Explanation\.?)\s*[.-]?\s*(.*)$', re.IGNORECASE | re.MULTILINE),
        
        # Illustration
        'illustration': re.compile(r'^(?:Illustration\s*[0-9]*[.:]?)\s*(.*)$', re.IGNORECASE | re.MULTILINE),
    }
    
    def __init__(self, act_name: str, jurisdiction: str = "central"):
        self.act_name = act_name
        self.jurisdiction = jurisdiction
        self.current_part = None
        self.current_chapter = None
        self.current_section = None
        
    def parse(self, text: str) -> list[Document]:
        """
        Parse document text into hierarchical chunks.
        
        Args:
            text: Raw legal document text
            
        Returns:
            List of LangChain Documents with metadata
        """
        documents = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or len(line) < 2:
                i += 1
                continue
                
            # Detect Act title (usually at the beginning)
            if i < 10 and self._is_act_title(line):
                # This might be the act name, skip as it's already in metadata
                i += 1
                continue
                
            # Detect Part
            part_match = self.PATTERNS['part'].match(line)
            if part_match and 'PRELIMINARY' not in line and len(line) < 100:
                self.current_part = part_match.group(1).strip()
                self.current_chapter = None
                self.current_section = None
                i += 1
                continue
                
            # Detect Chapter
            chapter_match = self.PATTERNS['chapter'].match(line)
            if chapter_match and len(line) < 100:
                self.current_chapter = chapter_match.group(1).strip()
                self.current_section = None
                i += 1
                continue
                
            # Detect Section - main pattern
            section_match = self.PATTERNS['section_main'].match(line)
            if section_match:
                section_num = section_match.group(1)
                section_title = section_match.group(2) or ""
                
                # Skip if this looks like a subsection number
                if len(section_num) > 3:
                    i += 1
                    continue
                    
                self.current_section = section_num
                
                # Collect section content
                section_content, end_idx = self._collect_section_content(lines, i + 1)
                
                # Create document for main section
                metadata = LegalChunkMetadata(
                    act_name=self.act_name,
                    part=self.current_part,
                    chapter=self.current_chapter,
                    section_number=section_num,
                    section_title=section_title.strip()[:200],
                    chunk_text=f"Section {section_num}: {section_title}\n{section_content}".strip(),
                    jurisdiction=self.jurisdiction,
                    **self._infer_metadata(section_num, section_content)
                )
                documents.append(Document(page_content=metadata.chunk_text, metadata=metadata.to_dict()))
                
                # Parse subsections within this section
                subsection_docs = self._parse_subsections(section_content, section_num)
                documents.extend(subsection_docs)
                
                i = end_idx
                continue
                
            # Fallback: numbered line that might be a section
            numbered_match = self.PATTERNS['section_numbered'].match(line)
            if numbered_match and self.current_section is None:
                section_num = numbered_match.group(1)
                section_title = numbered_match.group(2) or ""
                
                # Skip if too long (likely not a section header)
                if len(line) > 200:
                    i += 1
                    continue
                    
                self.current_section = section_num
                
                section_content, end_idx = self._collect_section_content(lines, i + 1)
                
                metadata = LegalChunkMetadata(
                    act_name=self.act_name,
                    part=self.current_part,
                    chapter=self.current_chapter,
                    section_number=section_num,
                    section_title=section_title.strip()[:200],
                    chunk_text=f"Section {section_num}: {section_title}\n{section_content}".strip(),
                    jurisdiction=self.jurisdiction,
                    **self._infer_metadata(section_num, section_content)
                )
                documents.append(Document(page_content=metadata.chunk_text, metadata=metadata.to_dict()))
                i = end_idx
                continue
                
            i += 1
            
        return documents
    
    def _is_act_title(self, line: str) -> bool:
        """Check if line is likely an act title."""
        line_upper = line.upper()
        return ('ACT' in line_upper or 'ACT,' in line) and len(line) < 150
    
    def _collect_section_content(self, lines: list, start_idx: int) -> tuple[str, int]:
        """Collect content belonging to current section until next section/part/chapter."""
        content_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Stop at next major structural element
            if self.PATTERNS['part'].match(line):
                break
            if self.PATTERNS['chapter'].match(line):
                break
            if self.PATTERNS['section_main'].match(line):
                break
            if self.PATTERNS['section_numbered'].match(line) and len(line) < 100:
                # Likely a new section
                break
                
            content_lines.append(line)
            i += 1
            
        return '\n'.join(content_lines), i
    
    def _parse_subsections(self, content: str, section_num: str) -> list[Document]:
        """Parse subsections within a section."""
        documents = []
        lines = content.split('\n')
        
        current_subsection = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            subsection_match = self.PATTERNS['subsection'].match(line)
            if subsection_match:
                # Save previous subsection
                if current_subsection and current_content:
                    metadata = LegalChunkMetadata(
                        act_name=self.act_name,
                        part=self.current_part,
                        chapter=self.current_chapter,
                        section_number=section_num,
                        subsection=current_subsection,
                        chunk_text=' '.join(current_content),
                        jurisdiction=self.jurisdiction,
                        **self._infer_metadata(section_num, ' '.join(current_content))
                    )
                    documents.append(Document(page_content=metadata.chunk_text, metadata=metadata.to_dict()))
                
                current_subsection = f"({subsection_match.group(1)})"
                current_content = [subsection_match.group(2)] if subsection_match.group(2) else []
            elif current_subsection:
                current_content.append(line)
        
        # Save last subsection
        if current_subsection and current_content:
            metadata = LegalChunkMetadata(
                act_name=self.act_name,
                part=self.current_part,
                chapter=self.current_chapter,
                section_number=section_num,
                subsection=current_subsection,
                chunk_text=' '.join(current_content),
                jurisdiction=self.jurisdiction,
                **self._infer_metadata(section_num, ' '.join(current_content))
            )
            documents.append(Document(page_content=metadata.chunk_text, metadata=metadata.to_dict()))
            
        return documents
    
    def _infer_metadata(self, section_num: str, content: str) -> dict:
        """
        Infer metadata from section content using heuristics.
        
        This is a key novelty module that automatically classifies:
        - Personal law applicability
        - Demographic scope
        - Law type (civil/criminal/procedural)
        """
        content_lower = content.lower()
        
        # Infer personal law applicability
        applicable_personal_law = []
        if any(term in content_lower for term in ['hindu', 'hindi', 'vedic', 'aryan', 'buddhist', 'jaina', 'sikh']):
            applicable_personal_law.append('hindu')
        if any(term in content_lower for term in ['muslim', 'mahomedan', 'islam', 'quran', 'nikah', 'mehr', 'talaq', 'iddat']):
            applicable_personal_law.append('muslim')
        if any(term in content_lower for term in ['christian', 'christ', 'bible', 'marriage act']):
            applicable_personal_law.append('christian')
        if any(term in content_lower for term in ['special marriage', 'inter-faith', 'inter-religion', 'secular', 'any person']):
            applicable_personal_law.append('secular')
        if not applicable_personal_law:
            applicable_personal_law = ['all']  # Default to all if no specific law detected
            
        # Infer demographic scope
        demographic_scope = []
        if any(term in content_lower for term in ['wife', 'married woman', 'husband', 'spouse', 'conjugal']):
            demographic_scope.append('married_woman')
        if any(term in content_lower for term in ['child', 'minor', 'infant', 'below 18', 'juvenile', 'adolescent']):
            demographic_scope.append('minor')
        if any(term in content_lower for term in ['working woman', 'employee', 'service', 'maternity', 'wages']):
            demographic_scope.append('working_woman')
        if any(term in content_lower for term in ['pregnant', 'maternity', 'unborn', 'conception']):
            demographic_scope.append('pregnant_woman')
        if any(term in content_lower for term in ['divorce', 'dissolution', 'separation', 'nullity', 'void']):
            demographic_scope.append('divorced_woman')
        if any(term in content_lower for term in ['widow', 'widower', 'deceased']):
            demographic_scope.append('widow')
        if not demographic_scope:
            demographic_scope.append('any')
            
        # Infer law type
        law_type = 'civil'
        if any(term in content_lower for term in ['offence', 'punishment', 'imprisonment', 'fine', 'cognizable', 'bail', 'arrest', 'non-bailable']):
            law_type = 'criminal'
        elif any(term in content_lower for term in ['procedure', 'application', 'filing', 'jurisdiction', 'appeal', 'court', 'magistrate', 'petition']):
            law_type = 'procedural'
            
        return {
            'applicable_personal_law': applicable_personal_law,
            'demographic_scope': demographic_scope,
            'law_type': law_type
        }


# =============================================================================
# CONFLICT MAP LOADER
# =============================================================================

class ConflictMapLoader:
    """Loads and manages known provision conflicts."""
    
    def __init__(self, conflict_map_path: str = "conflict_map.json"):
        self.conflict_map_path = conflict_map_path
        self.conflicts = []
        self._load()
        
    def _load(self):
        """Load conflict map from JSON file."""
        if os.path.exists(self.conflict_map_path):
            with open(self.conflict_map_path, 'r', encoding='utf-8') as f:
                self.conflicts = json.load(f)
        else:
            self.conflicts = []
            
    def get_conflicts(self) -> list:
        """Return list of known conflicts."""
        return self.conflicts
    
    def check_conflict(self, provision_a: str, provision_b: str) -> Optional[dict]:
        """Check if two provisions have a known conflict."""
        for conflict in self.conflicts:
            if (conflict['provision_a'] == provision_a and conflict['provision_b'] == provision_b) or \
               (conflict['provision_a'] == provision_b and conflict['provision_b'] == provision_a):
                return conflict
        return None


# =============================================================================
# PDF ACT EXTRACTOR
# =============================================================================

class PDFActExtractor:
    """
    Extracts acts from PDF files in the data directory.
    
    Combines PDF text extraction with hierarchical parsing
    and metadata enrichment.
    """
    
    def __init__(self, data_dir: str = "./data", jurisdiction: str = "central"):
        self.data_dir = data_dir
        self.jurisdiction = jurisdiction
        self.pdf_extractor = PDFTextExtractor(data_dir)
        self.act_name_extractor = ActNameExtractor()
        
    def extract_all_acts(self) -> list[tuple[str, str]]:
        """
        Extract all acts from PDFs.
        
        Returns:
            List of (act_name, text) tuples
        """
        acts = []
        documents = self.pdf_extractor.extract_all()
        
        for doc in documents:
            # Extract act name from text or metadata
            act_name, year = self.act_name_extractor.extract(doc.text, doc.filename)
            
            # Use metadata title if available
            if doc.metadata.get('title') and doc.metadata['title'] != doc.filename:
                act_name = doc.metadata['title']
                
            # Add year if found
            if year:
                if str(year) not in act_name:
                    act_name = f"{act_name}, {year}"
                    
            acts.append((act_name, doc.text))
            
        return acts


# =============================================================================
# INGESTION PIPELINE
# =============================================================================

class LegalDocumentIngestor:
    """
    Main ingestion pipeline orchestrator.
    
    Steps:
    1. Parse documents hierarchically from PDFs
    2. Generate embeddings using free HuggingFace model
    3. Store in ChromaDB with metadata
    """
    
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(self, persist_directory: str = "./chroma_db", data_dir: str = "./data"):
        self.persist_directory = persist_directory
        self.data_dir = data_dir
        
        # Initialize embeddings model (free, local)
        print(f"Loading embedding model: {self.EMBEDDING_MODEL}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={'device': 'gpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB vector store
        print(f"Initializing ChromaDB at {persist_directory}...")
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="legal_documents"
        )
        
        # Load conflict map
        self.conflict_loader = ConflictMapLoader()
        
        # PDF extractor
        self.pdf_extractor = PDFActExtractor(data_dir)
        
    def ingest_from_pdfs(self) -> int:
        """
        Ingest all PDFs from the data directory.
        
        Returns:
            Total number of chunks ingested
        """
        acts = self.pdf_extractor.extract_all_acts()
        
        if not acts:
            print("No acts extracted from PDFs. Check data directory.")
            return 0
            
        print(f"\nExtracted {len(acts)} act(s) from PDFs")
        
        total = 0
        for act_name, text in acts:
            count = self.ingest_document(text, act_name)
            total += count
            
        return total
        
    def ingest_document(self, text: str, act_name: str, jurisdiction: str = "central") -> int:
        """
        Ingest a single legal document.
        
        Args:
            text: Raw document text
            act_name: Name of the act
            jurisdiction: "central" or "state:<name>"
            
        Returns:
            Number of chunks ingested
        """
        # Parse document hierarchically
        parser = HierarchicalDocumentParser(act_name=act_name, jurisdiction=jurisdiction)
        documents = parser.parse(text)
        
        print(f"  Parsed {len(documents)} chunks from {act_name}")
        
        # Store in ChromaDB
        if documents:
            self.vectorstore.add_documents(documents)
            print(f"  Stored {len(documents)} chunks in ChromaDB")
            
        return len(documents)
    
    def ingest_multiple(self, documents: list[tuple[str, str, str]]) -> int:
        """
        Ingest multiple documents.
        
        Args:
            documents: List of (text, act_name, jurisdiction) tuples
            
        Returns:
            Total number of chunks ingested
        """
        total = 0
        for text, act_name, jurisdiction in documents:
            count = self.ingest_document(text, act_name, jurisdiction)
            total += count
        return total
    
    def get_vectorstore(self) -> Chroma:
        """Return the ChromaDB vector store for querying."""
        return self.vectorstore
    
    def clear_database(self):
        """Clear all documents from the database."""
        # Get the underlying collection and delete all
        client = self.vectorstore._client
        collection = client.get_collection("legal_documents")
        # Get all IDs and delete
        items = collection.get(include=[])
        if items['ids']:
            collection.delete(ids=items['ids'])
        print("Database cleared.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_ingestion(data_dir: str = "./data", clear_first: bool = False):
    """
    Run the complete ingestion pipeline from PDFs.
    
    Args:
        data_dir: Directory containing PDF files
        clear_first: Whether to clear existing database before ingestion
    """
    print(f"Data directory: {data_dir}")
    
    # Initialize ingestor
    ingestor = LegalDocumentIngestor(
        persist_directory="./chroma_db",
        data_dir=data_dir
    )
    
    # Clear database if requested
    if clear_first:
        print("\nClearing existing database...")
        ingestor.clear_database()
    
    # Ingest from PDFs
    total_chunks = ingestor.ingest_from_pdfs()

    print("INGESTION COMPLETE")
    print(f"Total chunks stored: {total_chunks}")
    print(f"Vector store location: ./chroma_db")
    
    return ingestor


if __name__ == "__main__":
    import sys
    
    # Allow specifying data directory as argument
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    clear_first = "--clear" in sys.argv
    
    run_ingestion(data_dir=data_dir, clear_first=clear_first)
