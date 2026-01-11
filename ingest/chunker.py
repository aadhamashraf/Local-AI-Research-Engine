"""
Semantic Chunker - LLM-based intelligent text chunking
"""

from typing import List, Dict, Any
from loguru import logger
import re

from llm.ollama_client import OllamaClient
from llm.prompts import SEMANTIC_CHUNKING_PROMPT, ENTITY_EXTRACTION_PROMPT


class SemanticChunker:
    """Chunk text into semantic units using LLM."""
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        target_chunk_size: int = 800,
        max_chunk_size: int = 1200,
        overlap: int = 100,
        use_llm: bool = True
    ):
        """
        Initialize semantic chunker.
        
        Args:
            ollama_client: Ollama client for LLM calls
            target_chunk_size: Target tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
            use_llm: Whether to use LLM for chunking (fallback to sentence-based)
        """
        self.client = ollama_client
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.use_llm = use_llm
    
    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into semantic units.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text or len(text.strip()) == 0:
            return []
        
        metadata = metadata or {}
        
        # Estimate token count (rough: 1 token ≈ 4 chars)
        estimated_tokens = len(text) // 4
        
        logger.info(f"Chunking text (~{estimated_tokens} tokens)")
        
        # If text is small enough, return as single chunk
        if estimated_tokens <= self.target_chunk_size:
            return [self._create_chunk(text, 0, metadata)]
        
        # Try LLM-based chunking first
        if self.use_llm:
            try:
                chunks = self._llm_chunk(text, metadata)
                if chunks:
                    logger.info(f"LLM chunking produced {len(chunks)} chunks")
                    return chunks
            except Exception as e:
                logger.warning(f"LLM chunking failed, falling back to sentence-based: {e}")
        
        # Fallback to sentence-based chunking
        chunks = self._sentence_chunk(text, metadata)
        logger.info(f"Sentence chunking produced {len(chunks)} chunks")
        return chunks
    
    def _llm_chunk(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Use LLM to chunk text semantically."""
        
        # For very long texts, split into manageable sections first
        max_llm_input = 4000  # chars
        
        if len(text) > max_llm_input:
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            current_section = []
            sections = []
            current_length = 0
            
            for para in paragraphs:
                if current_length + len(para) > max_llm_input and current_section:
                    sections.append('\n\n'.join(current_section))
                    current_section = [para]
                    current_length = len(para)
                else:
                    current_section.append(para)
                    current_length += len(para)
            
            if current_section:
                sections.append('\n\n'.join(current_section))
            
            # Chunk each section
            all_chunks = []
            for section in sections:
                section_chunks = self._llm_chunk_section(section, metadata)
                all_chunks.extend(section_chunks)
            
            return all_chunks
        else:
            return self._llm_chunk_section(text, metadata)
    
    def _llm_chunk_section(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk a single section using LLM."""
        
        prompt = SEMANTIC_CHUNKING_PROMPT.format(text=text)
        
        response = self.client.generate(
            prompt=prompt,
            temperature=0.1,  # Low temperature for consistent chunking
            max_tokens=4000
        )
        
        # Split by the marker
        chunk_texts = response.split('---CHUNK---')
        chunk_texts = [c.strip() for c in chunk_texts if c.strip()]
        
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunks.append(self._create_chunk(chunk_text, i, metadata))
        
        return chunks
    
    def _sentence_chunk(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback: chunk by sentences with target size."""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence) // 4  # Rough token estimate
            
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) // 4 for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        index: int,
        base_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update({
            "chunk_index": index,
            "chunk_size": len(text),
            "chunk_tokens": len(text) // 4  # Rough estimate
        })
        
        return {
            "content": text,
            "metadata": chunk_metadata
        }
    
    def extract_chunk_metadata(
        self,
        chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract additional metadata from chunk using LLM.
        
        Extracts entities, key concepts, etc.
        """
        content = chunk.get("content", "")
        
        try:
            prompt = ENTITY_EXTRACTION_PROMPT.format(text=content[:2000])  # Limit length
            
            response = self.client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse JSON response
            import json
            entities = json.loads(response)
            
            # Add to chunk metadata
            chunk["metadata"]["entities"] = entities
            
            logger.debug(f"Extracted entities: {entities}")
            
        except Exception as e:
            logger.warning(f"Failed to extract entities: {e}")
            chunk["metadata"]["entities"] = {}
        
        return chunk
