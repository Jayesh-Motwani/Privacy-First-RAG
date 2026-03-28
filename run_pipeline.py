"""
Legal RAG Pipeline - End-to-End Test Script
============================================
Demonstrates the complete pipeline from PDF ingestion to query processing.

Prerequisites:
1. Install dependencies: pip install -r requirements.txt
2. Install Ollama: https://ollama.ai
3. Pull model: ollama pull qwen2.5:7b

Usage:
    python run_pipeline.py
    python run_pipeline.py --data ./data --clear  # Re-ingest from PDFs
"""

import sys
import os
import argparse

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def check_ollama():
    """Check if Ollama is installed and running."""
    import subprocess
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Ollama is installed and running")
            # Check if required model is available
            if 'qwen2.5' in result.stdout or 'mistral' in result.stdout or 'llama' in result.stdout:
                print("✓ Required LLM model is available")
                return True
            else:
                print("⚠ Required model not found. Run: ollama pull qwen2.5:7b")
                return False
        return False
    except FileNotFoundError:
        print("✗ Ollama not found. Install from https://ollama.ai")
        return False
    except Exception as e:
        print(f"⚠ Ollama check failed: {e}")
        return False


def check_pdfs(data_dir: str) -> list:
    """Check for PDF files in the data directory."""
    from pathlib import Path
    
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    if pdf_files:
        print(f"✓ Found {len(pdf_files)} PDF file(s) in {data_dir}")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
    else:
        print(f"✗ No PDF files found in {data_dir}")
    return pdf_files


def run_ingestion(data_dir: str = "./data", clear_first: bool = False):
    """Run the document ingestion pipeline from PDFs."""
    print("\n" + "=" * 60)
    print("STEP 1: PDF DOCUMENT INGESTION")
    print("=" * 60)
    
    from ingest import run_ingestion
    ingestor = run_ingestion(data_dir=data_dir, clear_first=clear_first)
    return ingestor


def run_query_demo(data_dir: str = "./data"):
    """Run the query pipeline demonstration."""
    print("\n" + "=" * 60)
    print("STEP 2: QUERY PROCESSING DEMO")
    print("=" * 60)
    
    from query_pipeline import LegalQueryPipeline
    
    # Initialize pipeline
    pipeline = LegalQueryPipeline()
    
    # Demo queries based on actual Indian laws
    demo_queries = [
        {
            "query": "My husband beats me and doesn't give me money for food. What are my rights?",
            "profile": {
                "jurisdiction": "central",
                "personal_law": "hindu",
                "demographic": "married_woman"
            },
            "description": "Domestic violence and maintenance query"
        },
        {
            "query": "What is the procedure for filing an FIR for child abuse?",
            "profile": {
                "jurisdiction": "Karnataka",
                "personal_law": "all",
                "demographic": "minor"
            },
            "description": "POCSO and CrPC query"
        },
        {
            "query": "Can I get maintenance from my ex-husband for my children after divorce?",
            "profile": {
                "jurisdiction": "central",
                "personal_law": "hindu",
                "demographic": "divorced_woman"
            },
            "description": "Post-divorce maintenance query"
        },
        {
            "query": "What are my rights to stay in the matrimonial home during divorce proceedings?",
            "profile": {
                "jurisdiction": "central",
                "personal_law": "hindu",
                "demographic": "married_woman"
            },
            "description": "Right to residence under DV Act"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n{'='*60}")
        print(f"DEMO QUERY {i}: {demo['description']}")
        print(f"{'='*60}")
        print(f"\nUser Query: {demo['query']}")
        print(f"User Profile: {demo['profile']}")
        print("-" * 40)
        
        result = pipeline.process_query(demo['query'], demo['profile'])
        
        print(f"\n--- PIPELINE RESULTS ---")
        print(f"PII Detected: {result['pii_masked']}")
        if result['entities_found']:
            print(f"Entities Masked: {result['entities_found']}")
        print(f"\nRewritten Query (Legal Terminology):")
        print(f"  {result['rewritten_query']}")
        print(f"\nRetrieved {len(result['retrieved_chunks'])} legal provisions")
        
        if result['conflicts']:
            print(f"\n⚠ CONFLICTS DETECTED ({len(result['conflicts'])}):")
            for conflict in result['conflicts']:
                print(f"  - {conflict['provision_a']} vs {conflict['provision_b']}")
                print(f"    {conflict['note']}")
        
        print(f"\n--- GENERATED ANSWER ---")
        print(result['answer'])


def run_interactive_mode():
    """Run interactive query mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE QUERY MODE")
    print("=" * 60)
    print("Enter your queries below. Type 'quit' to exit.\n")
    
    from query_pipeline import LegalQueryPipeline
    
    pipeline = LegalQueryPipeline()
    
    # Get user profile once
    print("Set up your user profile (press Enter for defaults):")
    jurisdiction = input("  Jurisdiction [central]: ").strip() or "central"
    personal_law = input("  Personal Law [all]: ").strip() or "all"
    demographic = input("  Demographic [any]: ").strip() or "any"
    
    user_profile = {
        "jurisdiction": jurisdiction,
        "personal_law": personal_law,
        "demographic": demographic
    }
    
    print(f"\nProfile set: {user_profile}")
    print("-" * 40)
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not query:
                continue
                
            result = pipeline.process_query(query, user_profile)
            
            print(f"\n--- ANSWER ---")
            print(result['answer'])
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def list_documents(data_dir: str = "./data"):
    """List all PDF documents in the data directory."""
    from pathlib import Path
    from pdf_extractor import PDFTextExtractor
    
    print("\n" + "=" * 60)
    print("DOCUMENTS IN DATA DIRECTORY")
    print("=" * 60)
    
    extractor = PDFTextExtractor(data_dir)
    documents = extractor.extract_all()
    
    if not documents:
        print("No documents found or extraction failed.")
        return
        
    for doc in documents:
        print(f"\n📄 {doc.filename}")
        print(f"   Pages: {doc.page_count}")
        print(f"   Title: {doc.metadata.get('title', 'N/A')}")
        print(f"   Type: {doc.metadata.get('document_type', 'N/A')}")
        if doc.metadata.get('act_year'):
            print(f"   Year: {doc.metadata['act_year']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Legal RAG Pipeline - Privacy-First Indian Law Assistant"
    )
    parser.add_argument(
        "--data", "-d",
        default="./data",
        help="Directory containing PDF legal documents (default: ./data)"
    )
    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Clear existing ChromaDB before ingestion"
    )
    parser.add_argument(
        "--ingest-only", "-i",
        action="store_true",
        help="Run ingestion only, skip query demo"
    )
    parser.add_argument(
        "--query-only", "-q",
        action="store_true",
        help="Skip ingestion, run query demo only"
    )
    parser.add_argument(
        "--interactive", "-I",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--list-docs", "-l",
        action="store_true",
        help="List documents in data directory"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LEGAL RAG PIPELINE - PRIVACY-FIRST")
    print("Indian Family & Child Law Assistant")
    print("=" * 60)
    
    # Check for PDF files
    pdf_files = check_pdfs(args.data)
    
    # List documents if requested
    if args.list_docs:
        list_documents(args.data)
        if not args.ingest_only and not args.query_only and not args.interactive:
            return
    
    # Check Ollama
    ollama_ready = check_ollama()
    
    # Determine mode
    if args.ingest_only:
        # Ingestion only
        run_ingestion(data_dir=args.data, clear_first=args.clear)
        
    elif args.query_only:
        # Query demo only
        if ollama_ready:
            run_query_demo(data_dir=args.data)
        else:
            print("\n⚠ Ollama not ready")
            print("   Install Ollama and run: ollama pull qwen2.5:7b")
            
    elif args.interactive:
        # Interactive mode
        if ollama_ready:
            run_interactive_mode()
        else:
            print("\n⚠ Ollama not ready")
            print("   Install Ollama and run: ollama pull qwen2.5:7b")
            
    else:
        # Full demo (default)
        if pdf_files:
            run_ingestion(data_dir=args.data, clear_first=args.clear)
        else:
            print("\n⚠ No PDF files found. Skipping ingestion.")
            
        if ollama_ready:
            run_query_demo(data_dir=args.data)
        else:
            print("\n⚠ Skipping query demo - Ollama not ready")
            print("   Install Ollama and run: ollama pull qwen2.5:7b")


if __name__ == "__main__":
    main()
