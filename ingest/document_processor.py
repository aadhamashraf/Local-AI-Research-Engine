"""
Document Processor - Orchestrates the ingestion pipeline
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm import tqdm

from .pdf_loader import PDFLoader
from .text_cleaner import TextCleaner
from .chunker import SemanticChunker


class DocumentProcessor:
    """Process documents through the ingestion pipeline."""
    
    def __init__(
        self,
        chunker: SemanticChunker,
        supported_formats: List[str] = None
    ):
        """
        Initialize document processor.
        
        Args:
            chunker: Semantic chunker instance
            supported_formats: List of supported file extensions
        """
        self.chunker = chunker
        self.pdf_loader = PDFLoader()
        self.text_cleaner = TextCleaner()
        
        self.supported_formats = supported_formats or [
            ".pdf", ".txt", ".md", ".py", ".js", ".java", ".cpp", ".c", ".h"
        ]
    
    def process_file(
        self,
        file_path: str,
        extract_entities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a single file.
        
        Args:
            file_path: Path to file
            extract_entities: Whether to extract entities from chunks
            
        Returns:
            List of processed chunks
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        logger.info(f"Processing file: {path.name}")
        
        # Load document
        if path.suffix.lower() == ".pdf":
            doc_data = self.pdf_loader.load(str(path))
        else:
            doc_data = self._load_text_file(path)
        
        # Clean text
        cleaned_text = self.text_cleaner.clean(doc_data["text"])
        
        # Chunk text
        chunks = self.chunker.chunk(cleaned_text, doc_data["metadata"])
        
        # Extract entities if requested
        if extract_entities:
            logger.info("Extracting entities from chunks...")
            for chunk in tqdm(chunks, desc="Extracting entities"):
                self.chunker.extract_chunk_metadata(chunk)
        
        logger.info(f"Processed {path.name}: {len(chunks)} chunks")
        return chunks
    
    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
        extract_entities: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all supported files in a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            extract_entities: Whether to extract entities
            
        Returns:
            Dictionary mapping file paths to chunk lists
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all supported files
        files = []
        for ext in self.supported_formats:
            if recursive:
                files.extend(dir_path.rglob(f"*{ext}"))
            else:
                files.extend(dir_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} files to process")
        
        results = {}
        
        for file_path in tqdm(files, desc="Processing files"):
            try:
                chunks = self.process_file(str(file_path), extract_entities)
                results[str(file_path)] = chunks
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                continue
        
        total_chunks = sum(len(chunks) for chunks in results.values())
        logger.info(f"Processed {len(results)} files, {total_chunks} total chunks")
        
        return results
    
    def _load_text_file(self, path: Path) -> Dict[str, Any]:
        """Load a plain text or code file."""
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(path, 'r', encoding='latin-1') as f:
                text = f.read()
        
        metadata = {
            "source": path.name,
            "source_type": path.suffix[1:],  # Remove dot
            "file_path": str(path.absolute()),
            "title": path.stem
        }
        
        return {
            "text": text,
            "metadata": metadata
        }
    
    def get_statistics(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get statistics about processed chunks."""
        
        if not chunks:
            return {}
        
        total_chars = sum(len(c["content"]) for c in chunks)
        total_tokens = sum(c["metadata"].get("chunk_tokens", 0) for c in chunks)
        
        # Count entities
        all_entities = {}
        for chunk in chunks:
            entities = chunk["metadata"].get("entities", {})
            for entity_type, entity_list in entities.items():
                if entity_type not in all_entities:
                    all_entities[entity_type] = set()
                all_entities[entity_type].update(entity_list)
        
        entity_counts = {k: len(v) for k, v in all_entities.items()}
        
        return {
            "num_chunks": len(chunks),
            "total_characters": total_chars,
            "total_tokens": total_tokens,
            "avg_chunk_size": total_chars // len(chunks),
            "entities": entity_counts
        }
