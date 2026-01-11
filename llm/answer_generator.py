"""
Answer Generator - Synthesizes answers with citations
"""

from typing import List, Dict, Any, Tuple
from loguru import logger
import re
import json

from .ollama_client import OllamaClient
from .prompts import ANSWER_SYNTHESIS_PROMPT


class AnswerGenerator:
    """Generates cited answers from retrieved evidence."""
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        citation_format: str = "[{source} §{section}]"
    ):
        """
        Initialize answer generator.
        
        Args:
            ollama_client: Ollama client instance
            temperature: Generation temperature
            max_tokens: Maximum tokens in answer
            citation_format: Format string for citations
        """
        self.client = ollama_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.citation_format = citation_format
    
    def generate_answer(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate an answer with citations.
        
        Args:
            question: User's question
            evidence: List of evidence chunks with metadata
            stream: Whether to stream the response
            
        Returns:
            Dictionary with answer, citations, and confidence
        """
        # Format sources for the prompt
        formatted_sources = self._format_sources(evidence)
        
        # Create the prompt
        prompt = ANSWER_SYNTHESIS_PROMPT.format(
            question=question,
            sources=formatted_sources
        )
        
        logger.info(f"Generating answer for: {question}")
        
        if stream:
            return self._generate_streaming(prompt, evidence)
        else:
            return self._generate_complete(prompt, evidence)
    
    def _format_sources(self, evidence: List[Dict[str, Any]]) -> str:
        """Format evidence chunks for the prompt."""
        formatted = []
        
        for i, chunk in enumerate(evidence):
            source_id = f"source_{i+1}"
            source_name = chunk.get('metadata', {}).get('source', 'Unknown')
            section = chunk.get('metadata', {}).get('section', 'N/A')
            content = chunk.get('content', '')
            
            formatted.append(
                f"[{source_id}] {source_name} (Section: {section})\n{content}\n"
            )
        
        return "\n---\n".join(formatted)
    
    def _generate_complete(
        self,
        prompt: str,
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate complete answer (non-streaming)."""
        
        answer_text = self.client.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Extract citations
        citations = self._extract_citations(answer_text, evidence)
        
        # Calculate confidence
        confidence = self._calculate_confidence(answer_text, citations)
        
        return {
            "answer": answer_text,
            "citations": citations,
            "confidence": confidence,
            "num_sources": len(evidence)
        }
    
    def _generate_streaming(
        self,
        prompt: str,
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate streaming answer."""
        
        answer_chunks = []
        
        for chunk in self.client.stream_generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ):
            answer_chunks.append(chunk)
            yield chunk
        
        # After streaming is complete, process citations
        answer_text = "".join(answer_chunks)
        citations = self._extract_citations(answer_text, evidence)
        confidence = self._calculate_confidence(answer_text, citations)
        
        yield {
            "citations": citations,
            "confidence": confidence,
            "num_sources": len(evidence)
        }
    
    def _extract_citations(
        self,
        answer: str,
        evidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract citation references from the answer.
        
        Returns:
            List of citation objects with source info
        """
        citations = []
        
        # Pattern to match citations like [source_1 §Section]
        pattern = r'\[(source_\d+)(?:\s*§\s*([^\]]+))?\]'
        
        for match in re.finditer(pattern, answer):
            source_id = match.group(1)
            section = match.group(2) if match.group(2) else "N/A"
            
            # Extract source index
            try:
                idx = int(source_id.split('_')[1]) - 1
                if 0 <= idx < len(evidence):
                    chunk = evidence[idx]
                    citations.append({
                        "source_id": source_id,
                        "source_name": chunk.get('metadata', {}).get('source', 'Unknown'),
                        "section": section,
                        "content": chunk.get('content', '')[:200] + "...",  # Preview
                        "full_content": chunk.get('content', '')
                    })
            except (ValueError, IndexError):
                logger.warning(f"Invalid citation format: {source_id}")
        
        # Remove duplicates
        unique_citations = []
        seen = set()
        for citation in citations:
            key = (citation['source_id'], citation['section'])
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        logger.info(f"Extracted {len(unique_citations)} unique citations")
        return unique_citations
    
    def _calculate_confidence(
        self,
        answer: str,
        citations: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score for the answer.
        
        Based on:
        - Number of citations
        - Answer length
        - Presence of uncertainty phrases
        """
        confidence = 1.0
        
        # Check for uncertainty phrases
        uncertainty_phrases = [
            "not found in sources",
            "unclear",
            "uncertain",
            "may be",
            "might be",
            "possibly",
            "no information"
        ]
        
        answer_lower = answer.lower()
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                confidence *= 0.7
        
        # Reward citations
        if len(citations) == 0:
            confidence *= 0.3
        elif len(citations) < 2:
            confidence *= 0.6
        elif len(citations) >= 5:
            confidence = min(confidence * 1.2, 1.0)
        
        # Penalize very short answers
        if len(answer.split()) < 30:
            confidence *= 0.8
        
        return round(confidence, 2)
    
    def format_citation(
        self,
        source_name: str,
        section: str = "N/A"
    ) -> str:
        """Format a citation string."""
        return self.citation_format.format(
            source=source_name,
            section=section
        )
