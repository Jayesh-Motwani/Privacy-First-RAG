"""
Test script for PII BERT model integration.
Verifies that the fine-tuned PII model loads and works correctly.
"""

import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from query_pipeline import PIIMasker


def test_pii_masker():
    """Test the PII masker with sample queries."""
    print("=" * 60)
    print("PII MASKER TEST")
    print("=" * 60)
    
    # Initialize masker
    masker = PIIMasker()
    
    # Test cases with different PII types
    test_cases = [
        # Legal query with person names
        "My name is Rajesh Kumar and my husband Amit Kumar beats me.",
        
        # Query with location
        "I live at 123 MG Road, Bangalore, Karnataka and need legal help.",
        
        # Query with phone and Aadhaar
        "My phone number is 9876543210 and Aadhaar is 1234-5678-9012.",
        
        # Query with email
        "Contact me at rajesh.kumar@email.com for case details.",
        
        # Query with PAN
        "My PAN is ABCDE1234F and I need maintenance advice.",
        
        # Complex legal query with multiple PII
        "I am Sunita Devi, wife of Ramesh Singh from Delhi. He hasn't given me maintenance. His phone is 9812345678.",
        
        # Query with financial info
        "My account number is 123456789012 and I need money for legal fees.",
        
        # Query with DOB
        "My date of birth is 15 January 1990 and I was married in 2010.",
    ]
    
    print("\nTesting PII masking:\n")
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"--- Test {i} ---")
        print(f"Original: {test_text}")
        
        masked_text, entities = masker.mask(test_text)
        
        print(f"Masked:   {masked_text}")
        print(f"Entities: {entities}")
        print()
    
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_pii_masker()
