"""
PDF Text Extractor for Indian Legal Documents
=============================================
Extracts text from PDF files with special handling for:
- Indian court judgments
- Legislative acts and statutes
- Government notifications

Uses PyMuPDF (fitz) for robust PDF parsing with fallback to pypdf.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
    PymuPDF_AVAILABLE = True
except ImportError:
    PymuPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. Install with: pip install pymupdf")

try:
    from pypdf import PdfReader
    PyPDF_AVAILABLE = True
except ImportError:
    PyPDF_AVAILABLE = False
    print("Warning: pypdf not installed. Install with: pip install pypdf")


@dataclass
class ExtractedDocument:
    """Represents an extracted legal document."""
    filename: str
    filepath: str
    text: str
    metadata: dict
    page_count: int


class PDFTextExtractor:
    """
    Extracts text from PDF files with legal document optimizations.
    
    Features:
    - Preserves section numbering and hierarchy
    - Handles multi-column layouts
    - Removes page headers/footers
    - Extracts document metadata
    """
    
    # Patterns for detecting legal document types
    DOCUMENT_PATTERNS = {
        'act': re.compile(r'(?:ACT|Act)\s+(?:No\.?\s*)?(\d+)\s+(?:of\s+)?(\d{4})', re.IGNORECASE),
        'judgment': re.compile(r'(?:IN THE|BEFORE THE)\s+(?:SUPREME COURT|HIGH COURT|DISTRICT COURT)', re.IGNORECASE),
        'notification': re.compile(r'(?:NOTIFICATION|G\.?S\.?R\.?\s*\d+)', re.IGNORECASE),
    }
    
    # Common page header/footer patterns to remove
    HEADER_FOOTER_PATTERNS = [
        re.compile(r'^\s*\d+\s*$', re.MULTILINE),  # Standalone page numbers
        re.compile(r'^\s*Page\s*\d+\s*(?:of\s*\d+)?\s*$', re.MULTILINE),  # Page X of Y
        re.compile(r'^\s*\[\s*\d+\s*\]\s*$', re.MULTILINE),  # [1], [2] style
        re.compile(r'^\s*\(?\d+\)?\s*$', re.MULTILINE),  # (1), 1 style at line start
    ]
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the PDF extractor.
        
        Args:
            data_dir: Directory containing PDF files
        """
        self.data_dir = Path(data_dir)
        
    def extract_all(self) -> list[ExtractedDocument]:
        """
        Extract text from all PDFs in the data directory.
        
        Returns:
            List of ExtractedDocument objects
        """
        documents = []
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.data_dir}")
            return documents
            
        print(f"Found {len(pdf_files)} PDF file(s)")
        
        for pdf_path in pdf_files:
            try:
                doc = self.extract_pdf(pdf_path)
                if doc:
                    documents.append(doc)
                    print(f"  ✓ Extracted: {pdf_path.name} ({doc.page_count} pages)")
            except Exception as e:
                print(f"  ✗ Failed to extract {pdf_path.name}: {e}")
                
        return documents
    
    def extract_pdf(self, pdf_path: str | Path) -> Optional[ExtractedDocument]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractedDocument or None if extraction fails
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            print(f"File not found: {pdf_path}")
            return None
            
        # Try PyMuPDF first (better quality)
        if PymuPDF_AVAILABLE:
            return self._extract_with_pymupdf(pdf_path)
        elif PyPDF_AVAILABLE:
            return self._extract_with_pypdf(pdf_path)
        else:
            print("No PDF library available. Install pymupdf or pypdf.")
            return None
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> Optional[ExtractedDocument]:
        """Extract text using PyMuPDF (fitz)."""
        try:
            doc = fitz.open(pdf_path)
            full_text = []
            
            for page_num, page in enumerate(doc):
                # Extract text with layout preservation
                text = page.get_text("text", sort=True)
                
                # Clean up the text
                text = self._clean_page_text(text, page_num, len(doc))
                full_text.append(text)
            
            # Extract metadata
            metadata = self._extract_metadata(doc, pdf_path.name)
            
            doc.close()
            
            return ExtractedDocument(
                filename=pdf_path.name,
                filepath=str(pdf_path),
                text='\n'.join(full_text),
                metadata=metadata,
                page_count=len(doc)
            )
            
        except Exception as e:
            print(f"PyMuPDF extraction error for {pdf_path.name}: {e}")
            # Fallback to pypdf
            if PyPDF_AVAILABLE:
                return self._extract_with_pypdf(pdf_path)
            return None
    
    def _extract_with_pypdf(self, pdf_path: Path) -> Optional[ExtractedDocument]:
        """Extract text using pypdf (fallback)."""
        try:
            reader = PdfReader(pdf_path)
            full_text = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = self._clean_page_text(text, page_num, len(reader.pages))
                full_text.append(text)
            
            # Extract metadata
            metadata = self._extract_metadata_from_pypdf(reader, pdf_path.name)
            
            return ExtractedDocument(
                filename=pdf_path.name,
                filepath=str(pdf_path),
                text='\n'.join(full_text),
                metadata=metadata,
                page_count=len(reader.pages)
            )
            
        except Exception as e:
            print(f"pypdf extraction error for {pdf_path.name}: {e}")
            return None
    
    def _clean_page_text(self, text: str, page_num: int, total_pages: int) -> str:
        """
        Clean up extracted page text.
        
        Removes headers, footers, and page numbers while preserving
        legal document structure.
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines at start/end
            if not line:
                if cleaned_lines:  # Keep internal empty lines for structure
                    cleaned_lines.append('')
                continue
            
            # Skip page numbers and common headers/footers
            skip = False
            for pattern in self.HEADER_FOOTER_PATTERNS:
                if pattern.match(line):
                    skip = True
                    break
                    
            if not skip:
                cleaned_lines.append(line)
        
        # Remove excessive blank lines
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def _extract_metadata(self, doc, filename: str) -> dict:
        """Extract metadata from PyMuPDF document."""
        metadata = {
            'filename': filename,
            'title': '',
            'author': '',
            'subject': '',
            'document_type': 'unknown',
            'act_year': None,
            'act_number': None
        }
        
        # Try to get PDF metadata
        pdf_meta = doc.metadata or {}
        metadata['title'] = pdf_meta.get('title', '')
        metadata['author'] = pdf_meta.get('author', '')
        metadata['subject'] = pdf_meta.get('subject', '')
        
        # Analyze first few pages to detect document type
        sample_text = ""
        for i, page in enumerate(doc[:3]):  # First 3 pages
            sample_text += page.get_text("text")
            
        # Detect document type
        for doc_type, pattern in self.DOCUMENT_PATTERNS.items():
            if pattern.search(sample_text):
                metadata['document_type'] = doc_type
                break
                
        # Extract act number and year if present
        act_match = re.search(r'(?:ACT|Act)\s+(?:No\.?\s*)?(\d+)\s+(?:of\s+)?(\d{4})', sample_text)
        if act_match:
            metadata['act_number'] = act_match.group(1)
            metadata['act_year'] = act_match.group(2)
            
        # Try to get act name from title or filename
        if not metadata['title']:
            # Use filename as fallback
            metadata['title'] = self._infer_title_from_filename(filename)
            
        return metadata
    
    def _extract_metadata_from_pypdf(self, reader, filename: str) -> dict:
        """Extract metadata from pypdf reader."""
        metadata = {
            'filename': filename,
            'title': '',
            'author': '',
            'subject': '',
            'document_type': 'unknown',
            'act_year': None,
            'act_number': None
        }
        
        # Get PDF metadata
        pdf_meta = reader.metadata or {}
        metadata['title'] = pdf_meta.get('/Title', '')
        metadata['author'] = pdf_meta.get('/Author', '')
        metadata['subject'] = pdf_meta.get('/Subject', '')
        
        # Analyze first page for document type
        if reader.pages:
            sample_text = reader.pages[0].extract_text() or ""
            
            for doc_type, pattern in self.DOCUMENT_PATTERNS.items():
                if pattern.search(sample_text):
                    metadata['document_type'] = doc_type
                    break
                    
            act_match = re.search(r'(?:ACT|Act)\s+(?:No\.?\s*)?(\d+)\s+(?:of\s+)?(\d{4})', sample_text)
            if act_match:
                metadata['act_number'] = act_match.group(1)
                metadata['act_year'] = act_match.group(2)
                
        if not metadata['title']:
            metadata['title'] = self._infer_title_from_filename(filename)
            
        return metadata
    
    def _infer_title_from_filename(self, filename: str) -> str:
        """Infer document title from filename."""
        # Remove extension and clean up
        name = filename.replace('.pdf', '')
        
        # Replace underscores and hyphens with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        
        # Remove common prefixes
        name = re.sub(r'^showfile\s*\d*\s*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^a\d+\s*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^aa\d+\s*', '', name, flags=re.IGNORECASE)
        
        # Title case
        name = name.title()
        
        return name.strip()


class ActNameExtractor:
    """
    Extracts the official name of an Act from document text.
    
    Uses heuristics to find the act name in various formats:
    - "THE HINDU MARRIAGE ACT, 1955"
    - "Protection of Women from Domestic Violence Act, 2005"
    - "Code of Criminal Procedure, 1973"
    """
    
    PATTERNS = [
        # "THE XXX ACT, YEAR" format
        re.compile(r'^THE\s+([A-Z][A-Za-z\s]+)\s+ACT,\s*(\d{4})', re.MULTILINE),
        # "XXX Act, YEAR" format
        re.compile(r'^([A-Z][A-Za-z\s]+(?:Act|ACT))\s*,?\s*(\d{4})', re.MULTILINE),
        # "XXX Act, YEAR" with year first
        re.compile(r'(\d{4})\s*[,-]?\s*([A-Z][A-Za-z\s]+(?:Act|ACT))', re.MULTILINE),
        # Generic act name pattern
        re.compile(r'([A-Z][A-Za-z\s]+(?:Act|ACT)[^,.]*),?\s*(\d{4})', re.MULTILINE),
    ]
    
    def extract(self, text: str, filename: str = "") -> tuple[str, Optional[int]]:
        """
        Extract act name and year from text.
        
        Args:
            text: Document text
            filename: Optional filename for fallback
            
        Returns:
            Tuple of (act_name, year)
        """
        # Try each pattern
        for pattern in self.PATTERNS:
            match = pattern.search(text[:2000])  # Search first 2000 chars
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    # Has year
                    if groups[0].isdigit():
                        # Year is first
                        return groups[1].strip(), int(groups[0])
                    else:
                        # Name is first
                        return groups[0].strip(), int(groups[1])
                        
        # Fallback: use filename
        if filename:
            name = filename.replace('.pdf', '').replace('_', ' ').title()
            return name, None
            
        return "Unknown Act", None


def extract_pdfs(data_dir: str = "./data") -> list[ExtractedDocument]:
    """
    Convenience function to extract all PDFs from a directory.
    
    Args:
        data_dir: Directory containing PDF files
        
    Returns:
        List of ExtractedDocument objects
    """
    extractor = PDFTextExtractor(data_dir)
    return extractor.extract_all()


if __name__ == "__main__":
    # Test extraction
    print("=" * 60)
    print("PDF TEXT EXTRACTOR - TEST")
    print("=" * 60)
    
    extractor = PDFTextExtractor("./data")
    documents = extractor.extract_all()
    
    print(f"\nExtracted {len(documents)} document(s)\n")
    
    for doc in documents:
        print(f"--- {doc.filename} ---")
        print(f"Pages: {doc.page_count}")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Type: {doc.metadata.get('document_type', 'N/A')}")
        print(f"Text length: {len(doc.text)} chars")
        print(f"First 500 chars:\n{doc.text[:500]}...")
        print()
