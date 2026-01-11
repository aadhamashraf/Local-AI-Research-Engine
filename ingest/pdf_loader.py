"""
PDF Loader - Extract text and metadata from PDF files
"""

import pdfplumber
from typing import Dict, Any, List
from pathlib import Path
from loguru import logger
import re


class PDFLoader:
    """Load and extract content from PDF files."""
    
    def __init__(self):
        """Initialize PDF loader."""
        pass
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a PDF file and extract content.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with text, metadata, and structure
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.info(f"Loading PDF: {path.name}")
        
        try:
            with pdfplumber.open(path) as pdf:
                # Extract metadata
                metadata = self._extract_metadata(pdf, path)
                
                # Extract text with structure
                pages_content = []
                full_text = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        pages_content.append({
                            "page": page_num,
                            "text": text
                        })
                        full_text.append(text)
                
                combined_text = "\n\n".join(full_text)
                
                # Try to extract sections
                sections = self._extract_sections(combined_text)
                
                result = {
                    "text": combined_text,
                    "metadata": metadata,
                    "pages": pages_content,
                    "sections": sections,
                    "num_pages": len(pdf.pages)
                }
                
                logger.info(f"Extracted {len(combined_text)} characters from {len(pdf.pages)} pages")
                return result
                
        except Exception as e:
            logger.error(f"Failed to load PDF {path.name}: {e}")
            raise
    
    def _extract_metadata(self, pdf: Any, path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF."""
        metadata = {
            "source": path.name,
            "source_type": "pdf",
            "file_path": str(path.absolute())
        }
        
        # Extract PDF metadata if available
        if pdf.metadata:
            metadata.update({
                "title": pdf.metadata.get("Title", path.stem),
                "author": pdf.metadata.get("Author", "Unknown"),
                "creator": pdf.metadata.get("Creator", ""),
                "producer": pdf.metadata.get("Producer", ""),
                "creation_date": str(pdf.metadata.get("CreationDate", "")),
            })
        else:
            metadata["title"] = path.stem
            metadata["author"] = "Unknown"
        
        return metadata
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Attempt to extract sections from text based on headers.
        
        Common patterns:
        - 1. Introduction
        - 1 Introduction
        - I. Introduction
        - Introduction
        """
        sections = []
        
        # Pattern for numbered sections
        pattern = r'^(?:(?:\d+\.?|\b[IVX]+\.)\s+)?([A-Z][A-Za-z\s]+)$'
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            # Check if this looks like a header
            if len(line) < 100 and re.match(pattern, line):
                # Save previous section
                if current_section:
                    sections.append({
                        "title": current_section,
                        "content": "\n".join(current_content).strip()
                    })
                
                # Start new section
                current_section = line
                current_content = []
            else:
                if line:
                    current_content.append(line)
        
        # Add last section
        if current_section:
            sections.append({
                "title": current_section,
                "content": "\n".join(current_content).strip()
            })
        
        logger.debug(f"Extracted {len(sections)} sections")
        return sections
