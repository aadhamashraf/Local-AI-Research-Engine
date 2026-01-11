"""
Citation Mapper - Track sources and format citations
"""

from typing import List, Dict, Any
from loguru import logger


class CitationMapper:
    """Map chunks to sources and format citations."""
    
    def __init__(
        self,
        citation_format: str = "[{source} §{section}]"
    ):
        """
        Initialize citation mapper.
        
        Args:
            citation_format: Format string for citations
        """
        self.citation_format = citation_format
        self.source_map = {}  # Maps source IDs to metadata
        
        logger.info("Initialized citation mapper")
    
    def register_sources(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Register sources and assign citation IDs.
        
        Args:
            results: List of search results
            
        Returns:
            Results with citation IDs added
        """
        for i, result in enumerate(results):
            source_id = f"source_{i+1}"
            
            # Add citation ID to result
            result["citation_id"] = source_id
            
            # Store in map
            self.source_map[source_id] = {
                "source_name": result.get("metadata", {}).get("source", "Unknown"),
                "section": result.get("metadata", {}).get("section", "N/A"),
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {})
            }
        
        logger.debug(f"Registered {len(results)} sources")
        return results
    
    def format_citation(
        self,
        source_id: str,
        section: str = None
    ) -> str:
        """
        Format a citation string.
        
        Args:
            source_id: Source identifier
            section: Optional section override
            
        Returns:
            Formatted citation
        """
        if source_id not in self.source_map:
            return f"[{source_id}]"
        
        source_info = self.source_map[source_id]
        section = section or source_info.get("section", "N/A")
        
        return self.citation_format.format(
            source=source_info["source_name"],
            section=section
        )
    
    def get_bibliography(self) -> List[Dict[str, Any]]:
        """
        Generate a bibliography of all sources.
        
        Returns:
            List of source information
        """
        bibliography = []
        
        for source_id, info in self.source_map.items():
            bibliography.append({
                "id": source_id,
                "source": info["source_name"],
                "section": info.get("section", "N/A"),
                "preview": info.get("content", "")[:200] + "..."
            })
        
        return bibliography
    
    def get_source_content(
        self,
        source_id: str
    ) -> str:
        """Get full content for a source."""
        if source_id in self.source_map:
            return self.source_map[source_id].get("content", "")
        return ""
    
    def clear(self):
        """Clear all registered sources."""
        self.source_map = {}
        logger.debug("Cleared citation mapper")
