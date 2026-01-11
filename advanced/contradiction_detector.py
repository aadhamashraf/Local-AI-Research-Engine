"""
Contradiction Detector - Find conflicting information across sources
"""

from typing import List, Dict, Any
from loguru import logger
import json

from llm.ollama_client import OllamaClient
from llm.prompts import CONTRADICTION_DETECTION_PROMPT


class ContradictionDetector:
    """Detect contradictions between different sources."""
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize contradiction detector.
        
        Args:
            ollama_client: Ollama client instance
        """
        self.client = ollama_client
        logger.info("Initialized contradiction detector")
    
    def find_contradictions(
        self,
        results: List[Dict[str, Any]],
        topic: str = None
    ) -> List[Dict[str, Any]]:
        """
        Find contradictions among search results.
        
        Args:
            results: List of search results
            topic: Optional topic to focus on
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        logger.info(f"Checking for contradictions among {len(results)} sources")
        
        # Compare each pair of results
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                result1 = results[i]
                result2 = results[j]
                
                # Skip if from same source
                source1 = result1.get("metadata", {}).get("source", "")
                source2 = result2.get("metadata", {}).get("source", "")
                
                if source1 == source2:
                    continue
                
                # Check for contradiction
                contradiction = self._check_contradiction(
                    result1["content"],
                    result2["content"],
                    source1,
                    source2
                )
                
                if contradiction and contradiction.get("contradicts"):
                    contradictions.append({
                        "source1": source1,
                        "source2": source2,
                        "statement1": result1["content"][:200] + "...",
                        "statement2": result2["content"][:200] + "...",
                        "explanation": contradiction.get("explanation", ""),
                        "severity": contradiction.get("severity", "unknown")
                    })
        
        logger.info(f"Found {len(contradictions)} contradictions")
        return contradictions
    
    def _check_contradiction(
        self,
        statement1: str,
        statement2: str,
        source1: str,
        source2: str
    ) -> Dict[str, Any]:
        """Check if two statements contradict each other."""
        
        # Truncate if too long
        statement1 = statement1[:1000]
        statement2 = statement2[:1000]
        
        prompt = CONTRADICTION_DETECTION_PROMPT.format(
            source1=source1,
            statement1=statement1,
            source2=source2,
            statement2=statement2
        )
        
        try:
            response = self.client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse JSON response
            result = json.loads(response)
            return result
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse contradiction detection response")
            return {"contradicts": False}
        except Exception as e:
            logger.error(f"Error checking contradiction: {e}")
            return {"contradicts": False}
