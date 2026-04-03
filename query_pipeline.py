"""
Legal Query Pipeline
====================
Processes user queries through a privacy-first RAG pipeline with:
1. PII masking using local BERT NER model
2. Legal query rewriting using local Ollama LLM
3. Applicability filtering based on user profile
4. Semantic retrieval from ChromaDB
5. Conflict detection between retrieved provisions
6. LLM generation with RAG context

All models are free, local, and open-source. No paid APIs.
"""

import re
import json
import os
from dataclasses import dataclass
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class PIIMasker:
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    """
    Privacy-first PII masking using fine-tuned BERT NER model.
    
    Novelty: Uses local fine-tuned PII BERT model from pii_bert_model folder,
    with regex fallbacks for Indian ID numbers (Aadhaar, PAN, phone numbers).
    """
    
    # Path to fine-tuned PII BERT model
    PII_MODEL_PATH = "./pii_bert_model/checkpoint-26160"
    
    # Regex patterns for Indian ID numbers (fallback + additional coverage)
    PATTERNS = {
        'aadhaar': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
        'pan': re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),
        'phone': re.compile(r'\b(?:\+91[\s-]?)?\d{10}\b'),
        'passport': re.compile(r'\b[A-Z]\d{7}\b'),
        'voter_id': re.compile(r'\b[A-Z]{3}\d{7}\b'),
    }
    
    # Map model entity labels to generic placeholders for legal context
    LABEL_MAP = {
        # Person names
        'FIRSTNAME': '[PERSON]',
        'LASTNAME': '[PERSON]',
        'MIDDLENAME': '[PERSON]',
        'PREFIX': '[PERSON]',  # Mr., Mrs., etc.
        'USERNAME': '[PERSON]',
        
        # Locations
        'CITY': '[LOCATION]',
        'STATE': '[LOCATION]',
        'COUNTY': '[LOCATION]',
        'STREET': '[LOCATION]',
        'BUILDINGNUMBER': '[LOCATION]',
        'SECONDARYADDRESS': '[LOCATION]',
        'ZIPCODE': '[LOCATION]',
        'NEARBYGPSCOORDINATE': '[LOCATION]',
        'ORDINALDIRECTION': '[LOCATION]',
        
        # Organizations
        'COMPANYNAME': '[ORGANISATION]',
        'JOBAREA': '[ORGANISATION]',
        'JOBTYPE': '[ORGANISATION]',
        
        # Contact info
        'EMAIL': '[CONTACT]',
        'PHONEIMEI': '[PHONE]',
        'PHONENUMBER': '[PHONE]',
        'URL': '[URL]',
        'USERAGENT': '[DEVICE]',
        
        # Financial
        'ACCOUNTNAME': '[ACCOUNT]',
        'ACCOUNTNUMBER': '[ACCOUNT]',
        'BIC': '[ACCOUNT]',
        'IBAN': '[ACCOUNT]',
        'CREDITCARDNUMBER': '[CARD]',
        'CREDITCARDCVV': '[CARD]',
        'CREDITCARDISSUER': '[CARD]',
        'BITCOINADDRESS': '[CRYPTO]',
        'ETHEREUMADDRESS': '[CRYPTO]',
        'LITECOINADDRESS': '[CRYPTO]',
        'CURRENCY': '[MONEY]',
        'AMOUNT': '[MONEY]',
        'CURRENCYCODE': '[MONEY]',
        'CURRENCYNAME': '[MONEY]',
        'CURRENCYSYMBOL': '[MONEY]',
        
        # Identifiers
        'SSN': '[ID]',
        'PASSWORD': '[CREDENTIAL]',
        'PIN': '[CREDENTIAL]',
        'MAC': '[DEVICE]',
        'IP': '[DEVICE]',
        'IPV4': '[DEVICE]',
        'IPV6': '[DEVICE]',
        
        # Personal attributes
        'AGE': '[AGE]',
        'DOB': '[DOB]',
        'DATE': '[DATE]',
        'TIME': '[TIME]',
        'GENDER': '[GENDER]',
        'SEX': '[GENDER]',
        'EYECOLOR': '[ATTRIBUTE]',
        'HEIGHT': '[ATTRIBUTE]',
        'JOBTITLE': '[JOB]',
        
        # Vehicles
        'VEHICLEVIN': '[VEHICLE]',
        'VEHICLEVRM': '[VEHICLE]',
        
        # Masked numbers
        'MASKEDNUMBER': '[REDACTED]',
    }
    
    def __init__(self, model_path: str = None):
        """
        Initialize PII masker with fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned PII BERT model
        """
        self.model_path = model_path or self.PII_MODEL_PATH
        self.ner_pipeline = None
        
        print(f"Loading PII model from: {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=-1  # CPU
            )
            print("PII model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load PII model from {self.model_path}: {e}")
            print("Falling back to regex-only PII detection.")
        
    def mask(self, text: str) -> tuple[str, dict]:
        """
        Mask PII in the input text.
        
        Args:
            text: Input text potentially containing PII
            
        Returns:
            Tuple of (masked_text, entities_found)
        """
        entities_found = {}
        masked_text = text
        
        # Apply regex-based masking first (for Indian IDs)
        for entity_type, pattern in self.PATTERNS.items():
            matches = list(pattern.finditer(masked_text))
            if matches:
                entities_found[entity_type] = len(matches)
                masked_text = pattern.sub(
                    f'[{entity_type.upper()}]',
                    masked_text
                )
        
        # Apply NER-based masking using fine-tuned model
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(masked_text)
                
                # Sort by start position in reverse order to replace from end
                ner_results.sort(key=lambda x: x['start'], reverse=True)
                
                for entity in ner_results:
                    entity_group = entity.get('entity_group', '')
                    
                    # Skip 'O' (not an entity) and unknown labels
                    if entity_group == 'O' or not entity_group:
                        continue
                    
                    # Extract base label (remove B- or I- prefix)
                    base_label = entity_group.split('-')[-1] if '-' in entity_group else entity_group
                    
                    # Get placeholder from label map
                    placeholder = self.LABEL_MAP.get(base_label, '[REDACTED]')
                    
                    # Track entities found
                    if base_label not in entities_found:
                        entities_found[base_label] = 0
                    entities_found[base_label] += 1
                    
                    # Replace entity with placeholder
                    start, end = entity['start'], entity['end']
                    masked_text = masked_text[:start] + placeholder + masked_text[end:]
                    
            except Exception as e:
                print(f"NER processing error: {e}")
                
        return masked_text, entities_found


class LegalQueryRewriter:
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    """
    Rewrites layperson legal queries into precise legal terminology.
    
    Uses local Ollama LLM (qwen2.5:7b or mistral:7b) for query rewriting.
    No API calls to paid services.
    """
    
    REWRITE_PROMPT = PromptTemplate(
        input_variables=["masked_query"],
        template="""You are a legal terminology assistant for Indian women and child law.
Your task is to rewrite layperson queries into precise legal terminology.

Guidelines:
1. Identify the legal issue(s) in the query
3. Use proper legal terminology
4. Keep the query concise but legally precise
5. Return ONLY the rewritten query, no explanations

Examples:
- "My husband beats me" -> "What remedies are available under the Protection of Women from Domestic Violence Act, 2005 for physical abuse by husband?"
- "Can I get money from my ex-husband for the kids?" -> "What maintenance provisions are available under Section 125 CrPC and Hindu Adoptions and Maintenance Act for children after divorce?"
- "I want to adopt a child" -> "What is the adoption procedure under the Juvenile Justice Act, 2015 and Hindu Adoptions and Maintenance Act, 1956?"

Query: {masked_query}

Rewritten query:"""
    )
    
    def __init__(self, model_name: str = "qwen2.5:7b"):
        """
        Initialize the query rewriter with Ollama LLM.
        
        Args:
            model_name: Ollama model name (qwen2.5:7b or mistral:7b)
        """
        print(f"Initializing Ollama LLM: {model_name}...")
        try:
            self.llm = OllamaLLM(
                model=model_name,
                temperature=0.3,
                num_predict=256
            )
            print("Ollama LLM initialized successfully.")
        except Exception as e:
            print(f"Warning: Could not initialize Ollama LLM: {e}")
            print("Query rewriting will use fallback mode.")
            self.llm = None
            
    def rewrite(self, masked_query: str) -> str:
        """
        Rewrite a masked query into legal terminology.
        
        Args:
            masked_query: Query with PII already masked
            
        Returns:
            Rewritten query in legal terminology
        """
        if self.llm is None:
            # Fallback: return original query with minor cleanup
            return masked_query.strip()
            
        try:
            prompt = self.REWRITE_PROMPT.format(masked_query=masked_query)
            rewritten = self.llm.invoke(prompt)
            return rewritten.strip()
        except Exception as e:
            print(f"Query rewriting error: {e}")
            return masked_query.strip()


@dataclass
class UserProfile:
    """User profile for filtering applicable legal provisions."""
    jurisdiction: str = "central"  # e.g., "Karnataka", "central"
    personal_law: str = "all"  # "hindu", "muslim", "christian", "secular", "all"
    demographic: str = "any"  # "married_woman", "minor", "working_woman", "any"
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserProfile':
        """Create UserProfile from dictionary."""
        return cls(
            jurisdiction=data.get('jurisdiction', 'central'),
            personal_law=data.get('personal_law', 'all'),
            demographic=data.get('demographic', 'any')
        )


class ApplicabilityFilter:
    """
    Builds ChromaDB filters based on user profile.

    Novelty: Enables filtering legal provisions by:
    - Jurisdiction (central vs state laws)
    - Personal law applicability (Hindu, Muslim, Christian, secular)
    - Demographic scope (married women, minors, working women, etc.)
    """

    def build_filter(self, profile: UserProfile) -> Optional[dict]:
        """
        Build ChromaDB where filter from user profile.

        Args:
            profile: User profile with jurisdiction, personal_law, demographic

        Returns:
            ChromaDB filter dict or None if no filtering needed
        """
        conditions = []

        # Personal law filter - use $in to check if the comma-separated string contains the value
        # Since metadata is stored as comma-separated strings, we check for exact matches
        # or strings that contain the value as part of a comma-separated list
        if profile.personal_law and profile.personal_law != 'all':
            # Build a regex-like pattern match using $in with possible positions
            # Value could be: "hindu", "all,hindu", "hindu,muslim", "all,hindu,muslim", etc.
            # For simplicity, we'll retrieve all and let semantic search handle relevance
            # ChromaDB doesn't support CONTAINS for strings, so we skip this filter
            pass

        # Demographic filter - same issue as personal law
        if profile.demographic and profile.demographic != 'any':
            # Skip filter due to ChromaDB limitations with string contains
            pass

        # Jurisdiction filter (exact match)
        if profile.jurisdiction and profile.jurisdiction != 'central':
            conditions.append({
                "jurisdiction": {"$in": [profile.jurisdiction, "central"]}
            })

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$or": conditions}


class ConflictDetector:
    """
    Detects conflicts between retrieved legal provisions.
    
    Loads known conflicts from conflict_map.json and checks
    if any retrieved provisions have known tensions.
    """
    
    def __init__(self, conflict_map_path: str = "conflict_map.json"):
        self.conflict_map = self._load_conflict_map(conflict_map_path)
        
    def _load_conflict_map(self, path: str) -> list:
        """Load conflict map from JSON file."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
        
    def get_provision_id(self, doc: Document) -> str:
        """Extract provision ID from document metadata."""
        act_name = doc.metadata.get('act_name', '')
        section_num = doc.metadata.get('section_number', '')
        
        # Create abbreviated act name
        act_abbr = ''.join(word[0] for word in act_name.split() if word[0].isalpha())
        act_abbr = act_abbr.upper()[:6]  # Max 6 chars
        
        if section_num:
            return f"{act_abbr}_Section_{section_num}"
        return act_abbr
        
    def detect_conflicts(self, documents: list[Document]) -> list[dict]:
        """
        Check for conflicts between retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of detected conflicts with notes
        """
        detected = []
        provision_ids = [self.get_provision_id(doc) for doc in documents]
        
        # Check all pairs
        for i, prov_a in enumerate(provision_ids):
            for prov_b in provision_ids[i+1:]:
                for conflict in self.conflict_map:
                    if (conflict['provision_a'] == prov_a and conflict['provision_b'] == prov_b) or \
                       (conflict['provision_a'] == prov_b and conflict['provision_b'] == prov_a):
                        detected.append({
                            'provision_a': prov_a,
                            'provision_b': prov_b,
                            'note': conflict['note']
                        })
                        
        return detected
        
    def format_conflict_notice(self, conflicts: list[dict]) -> str:
        """Format detected conflicts as a notice string for the LLM."""
        if not conflicts:
            return ""
            
        notice = "**LEGAL CONFLICT NOTICE:**\n"
        notice += "The following provisions may have conflicting interpretations:\n\n"
        
        for conflict in conflicts:
            notice += f"- {conflict['provision_a']} vs {conflict['provision_b']}\n"
            notice += f"  Note: {conflict['note']}\n\n"
            
        notice += "Please consider these tensions in your response.\n"
        return notice


class LegalQueryPipeline:
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    """
    Complete query pipeline orchestrator.
    
    Coordinates all pipeline stages:
    1. PII masking
    2. Query rewriting
    3. Applicability filtering
    4. Semantic retrieval
    5. Conflict detection
    6. LLM generation
    """
    

    GENERATION_PROMPT = PromptTemplate(
        input_variables=["conflict_notice", "context", "question"],
        template="""You are a legal aid assistant helping women and children understand their legal rights under Indian law.

IMPORTANT GUIDELINES:
1. Answer based ONLY on the provided legal context
2. Cite specific section numbers and act names where possible
3. Use clear, accessible language while maintaining legal accuracy
4. If the context doesn't contain enough information, say so
5. Do not provide legal advice - only explain what the law says
6. Be sensitive to the vulnerable position of women and children seeking help

{conflict_notice}

LEGAL CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    )
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        llm_model: str = "qwen2.5:7b",
        k_retrievals: int = 5
    ):
        """
        Initialize the query pipeline.
        
        Args:
            persist_directory: ChromaDB persistence directory
            llm_model: Ollama model name for generation
            k_retrievals: Number of chunks to retrieve
        """
        self.k_retrievals = k_retrievals
        
        # Initialize components
        print("Initializing Query Pipeline...")
        
        # PII Masker
        self.pii_masker = PIIMasker()
        
        # Query Rewriter
        self.query_rewriter = LegalQueryRewriter(model_name=llm_model)
        
        # Embeddings and Vector Store
        print(f"Loading embeddings model: {self.EMBEDDING_MODEL}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print(f"Loading ChromaDB from {persist_directory}...")
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="legal_documents"
        )
        
        # Applicability Filter
        self.applicability_filter = ApplicabilityFilter()
        
        # Conflict Detector
        self.conflict_detector = ConflictDetector()
        
        # LLM for generation
        print(f"Initializing Ollama LLM for generation: {llm_model}...")
        try:
            self.llm = OllamaLLM(
                model=llm_model,
                temperature=0.2,
                num_predict=1024
            )
            print("LLM initialized successfully.")
        except Exception as e:
            print(f"Warning: Could not initialize LLM: {e}")
            self.llm = None
            
        print("Query Pipeline ready.")
        
    def process_query(self, user_query: str, user_profile: dict) -> dict:
        """
        Process a user query through the complete pipeline.
        
        Args:
            user_query: Raw user query (may contain PII)
            user_profile: Dict with jurisdiction, personal_law, demographic
            
        Returns:
            Dict with:
            - answer: LLM-generated response
            - rewritten_query: Legal terminology version
            - retrieved_chunks: Retrieved documents with metadata
            - conflicts: Detected provision conflicts
            - pii_masked: Whether PII was detected and masked
        """
        profile = UserProfile.from_dict(user_profile)
        result = {
            'original_query': user_query,
            'rewritten_query': '',
            'answer': '',
            'retrieved_chunks': [],
            'conflicts': [],
            'pii_masked': False,
            'entities_found': {}
        }
        
        # Step 1: PII Masking
        print("\n[Step 1] Masking PII...")
        masked_query, entities_found = self.pii_masker.mask(user_query)
        result['entities_found'] = entities_found
        result['pii_masked'] = len(entities_found) > 0
        print(f"  Entities found: {entities_found}")
        
        # Step 2: Query Rewriting
        print("\n[Step 2] Rewriting query...")
        rewritten_query = self.query_rewriter.rewrite(masked_query)
        result['rewritten_query'] = rewritten_query
        print(f"  Rewritten: {rewritten_query[:100]}...")
        
        # Step 3: Build Applicability Filter
        print("\n[Step 3] Building applicability filter...")
        db_filter = self.applicability_filter.build_filter(profile)
        print(f"  Filter: {db_filter}")
        
        # Step 4: Semantic Retrieval
        print("\n[Step 4] Retrieving relevant chunks...")
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.k_retrievals,
                "filter": db_filter
            } if db_filter else {"k": self.k_retrievals}
        )
        
        try:
            retrieved_docs = retriever.invoke(rewritten_query)
        except Exception as e:
            print(f"Retrieval error: {e}")
            retrieved_docs = []
            
        result['retrieved_chunks'] = [
            {
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in retrieved_docs
        ]
        print(f"  Retrieved {len(retrieved_docs)} chunks")
        
        # Step 5: Conflict Detection
        print("\n[Step 5] Checking for provision conflicts...")
        conflicts = self.conflict_detector.detect_conflicts(retrieved_docs)
        result['conflicts'] = conflicts
        conflict_notice = self.conflict_detector.format_conflict_notice(conflicts)
        print(f"  Conflicts detected: {len(conflicts)}")
        
        # Step 6: LLM Generation
        print("\n[Step 6] Generating response...")
        
        # Format context with metadata
        context = ""
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.metadata
            context += f"\n[Source {i}]\n"
            context += f"Act: {meta.get('act_name', 'N/A')}\n"
            context += f"Section: {meta.get('section_number', 'N/A')}\n"
            if meta.get('section_title'):
                context += f"Title: {meta.get('section_title')}\n"
            context += f"Content: {doc.page_content}\n"
            context += "-" * 40 + "\n"
            
        if not context:
            context = "No relevant legal provisions found in the database."
            
        # Generate response
        if self.llm:
            try:
                prompt = self.GENERATION_PROMPT.format(
                    conflict_notice=conflict_notice if conflict_notice else "",
                    context=context,
                    question=rewritten_query
                )
                answer = self.llm.invoke(prompt)
                result['answer'] = answer.strip()
            except Exception as e:
                print(f"Generation error: {e}")
                result['answer'] = "Unable to generate response. Please ensure Ollama is running with the required model."
        else:
            result['answer'] = "LLM not available. Please ensure Ollama is installed and running with qwen2.5:7b or mistral:7b model."
            
        print("\n" + "=" * 60)
        print("QUERY PROCESSING COMPLETE")
        print("=" * 60)
        
        return result


def process_query(user_query: str, user_profile: dict) -> dict:
    """
    Convenience function to process a query through the pipeline.
    
    Args:
        user_query: Raw user query
        user_profile: Dict with keys:
            - jurisdiction: str (e.g., "Karnataka", "central")
            - personal_law: str (e.g., "hindu", "muslim", "all")
            - demographic: str (e.g., "married_woman", "minor", "any")
            
    Returns:
        Dict with answer, rewritten_query, retrieved_chunks, conflicts, pii_masked
    """
    pipeline = LegalQueryPipeline()
    return pipeline.process_query(user_query, user_profile)


if __name__ == "__main__":
    # Test the pipeline
    print("=" * 60)
    print("LEGAL QUERY PIPELINE - TEST MODE")
    print("=" * 60)
    
    # Sample queries for testing
    test_queries = [
        {
            "query": "My husband beats me and doesn't give me money. What can I do?",
            "profile": {
                "jurisdiction": "central",
                "personal_law": "hindu",
                "demographic": "married_woman"
            }
        },
        {
            "query": "I want to adopt a child. I am Hindu. What is the process?",
            "profile": {
                "jurisdiction": "central",
                "personal_law": "hindu",
                "demographic": "any"
            }
        },
        {
            "query": "Can I get maintenance from my ex-husband for my children?",
            "profile": {
                "jurisdiction": "Karnataka",
                "personal_law": "hindu",
                "demographic": "divorced_woman"
            }
        }
    ]
    
    # Initialize pipeline
    pipeline = LegalQueryPipeline()
    
    for i, test in enumerate(test_queries, 1):
        print(f"Query: {test['query']}")
        print(f"Profile: {test['profile']}")
        
        result = pipeline.process_query(test['query'], test['profile'])

        print(f"PII Masked: {result['pii_masked']}")
        print(f"Entities Found: {result['entities_found']}")
        print(f"Rewritten Query: {result['rewritten_query']}")
        print(f"Retrieved Chunks: {len(result['retrieved_chunks'])}")
        print(f"Conflicts Detected: {len(result['conflicts'])}")
        print(f"\nANSWER:\n{result['answer']}")
