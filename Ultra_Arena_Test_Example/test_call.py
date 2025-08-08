#!/usr/bin/env python3
"""
Test call script to process a single German PDF file using Ultra_Arena_Main
"""

import sys
import json
from pathlib import Path

# Add the Ultra_Arena_Main directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "Ultra_Arena_Main"))

from main_modular import run_file_processing_simple

def main():
    # Define the input directory and specific file path
    input_dir = Path(__file__).parent / "input_files" / "1_file"
    pdf_file_path = input_dir / "german-sample-1.pdf"
    
    # Call the simple processing function
    results = run_file_processing_simple(
        input_pdf_dir_path=input_dir,
        pdf_file_paths=[pdf_file_path]
    )
    
    # Print results in prettified JSON format
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
