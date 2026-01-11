"""
LLM Reranker - Rerank results using LLM relevance scoring
"""

from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm

from llm.ollama_client import OllamaClient
from llm.prompts import RERANKING_PROMPT


class LLMReranker:
    """Rerank search results using LLM."""
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        temperature: float = 0.1
    ):
        """
        Initialize LLM reranker.
        
        Args:
            ollama_client: Ollama client instance
            temperature: Generation temperature
        """
        self.client = ollama_client
        self.temperature = temperature
        
        logger.info("Initialized LLM reranker")
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using LLM relevance scoring.
        
        Args:
            query: Original query
            results: List of search results
            top_k: Number of top results to return
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        logger.info(f"Reranking {len(results)} results")
        
        # Score each result
        scored_results = []
        
        for result in tqdm(results, desc="Reranking"):
            score = self._score_relevance(query, result)
            result_copy = result.copy()
            result_copy["llm_relevance_score"] = score
            scored_results.append(result_copy)
        
        # Sort by LLM score
        sorted_results = sorted(
            scored_results,
            key=lambda x: x.get("llm_relevance_score", 0),
            reverse=True
        )
        
        logger.info(f"Reranking complete, returning top {top_k}")
        return sorted_results[:top_k]
    
    def _score_relevance(
        self,
        query: str,
        result: Dict[str, Any]
    ) -> float:
        """
        Score the relevance of a result to the query.
        
        Returns:
            Relevance score (0-10)
        """
        passage = result.get("content", "")
        
        # Truncate if too long
        if len(passage) > 1000:
            passage = passage[:1000] + "..."
        
        prompt = RERANKING_PROMPT.format(
            query=query,
            passage=passage
        )
        
        try:
            response = self.client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=10
            )
            
            # Extract number from response
            score = self._extract_score(response)
            logger.debug(f"Relevance score: {score}")
            return score
            
        except Exception as e:
            logger.warning(f"Failed to score relevance: {e}")
            return 0.0
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from LLM response."""
        import re
        
        # Try to find a number
        matches = re.findall(r'\d+\.?\d*', response)
        
        if matches:
            try:
                score = float(matches[0])
                # Clamp to 0-10
                return max(0.0, min(10.0, score))
            except ValueError:
                pass
        
        # Default to 5 if can't parse
        logger.warning(f"Could not parse score from: {response}")
        return 5.0
