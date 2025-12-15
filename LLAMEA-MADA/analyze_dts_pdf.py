#!/usr/bin/env python3
"""Extract and analyze D-TS PDF to compare with implementation."""

import sys
import re

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    try:
        import pdfplumber
        HAS_PDFPLUMBER = True
    except ImportError:
        HAS_PDFPLUMBER = False

def extract_pdf_text(pdf_path):
    """Extract text from PDF."""
    if HAS_PYPDF2:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    elif HAS_PDFPLUMBER:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text
    else:
        return None

def find_dts_algorithm(text):
    """Find D-TS algorithm description."""
    # Look for algorithm descriptions, update rules, formulas
    patterns = [
        r'[Nn]_{?i}?\s*[←=]\s*[γγ]?\s*[Nn]_{?i}?',
        r'[μμ]̃_{?i}?\s*[←=]\s*[γγ]?\s*[μμ]̃_{?i}?',
        r'[ττ]_{?i}?\s*[←=]\s*.*sqrt',
        r'[μμ]̂_{?i}?\s*[←=]\s*[μμ]̃.*[Nn]',
        r'discount.*factor',
        r'Algorithm\s*\d+',
        r'Update.*rule',
    ]
    
    results = {}
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            results[pattern] = matches[:5]  # First 5 matches
    
    return results

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "D-TS.pdf")
    
    print("Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    
    if text is None:
        print("ERROR: Need PyPDF2 or pdfplumber to read PDF")
        print("Install with: pip install PyPDF2")
        sys.exit(1)
    
    print(f"Extracted {len(text)} characters")
    
    # Save to file for analysis
    output_file = os.path.join(script_dir, "D-TS_extracted.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved extracted text to: {output_file}")
    
    print("\n" + "="*80)
    print("First 2000 characters (UTF-8 safe):")
    print("="*80)
    try:
        print(text[:2000].encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
    except:
        print(text[:2000])
    
    print("\n" + "="*80)
    print("Searching for D-TS algorithm patterns:")
    print("="*80)
    results = find_dts_algorithm(text)
    for pattern, matches in results.items():
        print(f"\nPattern: {pattern}")
        for match in matches:
            print(f"  - {match}")
    
    # Look for specific formulas
    print("\n" + "="*80)
    print("Key sections (searching for 'update', 'discount', 'tau', 'sigma'):")
    print("="*80)
    
    keywords = ['update', 'discount', 'tau', 'sigma', 'N_i', 'mu', 'algorithm']
    for keyword in keywords:
        # Find context around keyword
        pattern = f'.{{0,100}}{re.escape(keyword)}.{{0,100}}'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            print(f"\n--- Found '{keyword}' ---")
            for match in matches[:2]:  # First 2 contexts
                print(f"  {match.strip()}")
