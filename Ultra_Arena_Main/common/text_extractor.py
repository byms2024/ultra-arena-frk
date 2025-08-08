"""
Text extraction module supporting multiple PDF extraction libraries.
"""

import os
import logging
import regex as re
from pathlib import Path
from typing import Dict, Any, Optional

# Import PDF extraction libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. Install with: pip install PyMuPDF")

try:
    import pytesseract
    from PIL import Image
    import pdf2image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logging.warning("Pytesseract not available. Install with: pip install pytesseract pdf2image Pillow")


class TextExtractor:
    """Text extraction from PDFs using multiple libraries."""
    
    def __init__(self, extractor_lib: str = "pymupdf"):
        self.extractor_lib = extractor_lib.lower()
        
        if self.extractor_lib == "pymupdf" and not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not available")
        elif self.extractor_lib == "pytesseract" and not PYTESSERACT_AVAILABLE:
            raise ImportError("Pytesseract not available")
    
    def extract_text(self, pdf_path: str, max_length: int = 50000) -> str:
        """Extract text from PDF using the specified library."""
        if self.extractor_lib == "pymupdf":
            return self._extract_with_pymupdf(pdf_path, max_length)
        elif self.extractor_lib == "pytesseract":
            return self._extract_with_pytesseract(pdf_path, max_length)
        else:
            raise ValueError(f"Unsupported extractor library: {self.extractor_lib}")
    
    def _extract_with_pymupdf(self, pdf_path: str, max_length: int) -> str:
        """Extract text using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += page_text + "\n"
                
                # Check if we've exceeded max length
                if len(text) > max_length:
                    text = text[:max_length]
                    break
            
            doc.close()
            return self._clean_text(text)
            
        except Exception as e:
            logging.error(f"PyMuPDF extraction error for {pdf_path}: {e}")
            return ""
    
    def _extract_with_pytesseract(self, pdf_path: str, max_length: int) -> str:
        """Extract text using Pytesseract OCR."""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path)
            text = ""
            
            for i, image in enumerate(images):
                # Extract text from image using OCR
                page_text = pytesseract.image_to_string(image, lang='por')
                text += page_text + "\n"
                
                # Check if we've exceeded max length
                if len(text) > max_length:
                    text = text[:max_length]
                    break
            
            return self._clean_text(text)
            
        except Exception as e:
            logging.error(f"Pytesseract extraction error for {pdf_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text for LLM processing."""
        # Remove excessive whitespace, newlines, and control characters
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-printable ASCII characters
        text = ''.join(char for char in text if 31 < ord(char) < 127 or ord(char) in [10, 13])
        return text
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF."""
        try:
            if self.extractor_lib == "pymupdf":
                return self._extract_metadata_pymupdf(pdf_path)
            else:
                # For pytesseract, we can't extract metadata easily
                return {"pages": 0, "size_mb": 0}
        except Exception as e:
            logging.error(f"Metadata extraction error for {pdf_path}: {e}")
            return {"pages": 0, "size_mb": 0}
    
    def _extract_metadata_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            metadata = {
                "pages": len(doc),
                "size_mb": os.path.getsize(pdf_path) / (1024 * 1024),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", "")
            }
            doc.close()
            return metadata
        except Exception as e:
            logging.error(f"PyMuPDF metadata extraction error: {e}")
            return {"pages": 0, "size_mb": 0}


class RegexExtractor:
    """Extract specific fields using regex patterns."""
    
    @staticmethod
    def extract_doc_type(text: str) -> Optional[str]:
        """Extract document type using regex."""
        text_upper = text.upper()
        
        # Check for service indicators
        service_indicators = ["NFS-E", "NFSE", "NOTA FISCAL ELETRÔNICA DE SERVIÇOS", "TOMADOR DE SERVIÇOS"]
        for indicator in service_indicators:
            if indicator in text_upper:
                return "Serviço"
        
        # Check for parts indicators
        if "NF-E" in text_upper:
            return "Peças"
        
        return "Outros"
    
    @staticmethod
    def extract_claim_number(text: str) -> Optional[str]:
        pattern = r''
        match = re.search(pattern, text)
        return match.group(0) if match else None
    
    @staticmethod
    def extract_chassi(text: str) -> Optional[str]:
        pattern = r''
        match = re.search(pattern, text)
        return match.group(0) if match else None
    
    @staticmethod
    def extract_cnpj(text: str) -> Optional[str]:
        """Extract CNPJ using regex."""
        # Pattern for CNPJ format XX.XXX.XXX/XXXX-XX
        pattern = r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}'
        match = re.search(pattern, text)
        return match.group(0) if match else None
    
    @staticmethod
    def extract_all_fields(text: str) -> Dict[str, Optional[str]]:
        return {} 