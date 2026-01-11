"""
Literature Review Generator - Auto-generate structured literature reviews
"""

from typing import List, Dict, Any
from loguru import logger
from collections import defaultdict

from llm.ollama_client import OllamaClient
from llm.prompts import LITERATURE_REVIEW_PROMPT


class LiteratureReviewGenerator:
    """Generate literature reviews from multiple sources."""
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize literature review generator.
        
        Args:
            ollama_client: Ollama client instance
        """
        self.client = ollama_client
        logger.info("Initialized literature review generator")
    
    def generate_review(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        max_sources: int = 10
    ) -> Dict[str, Any]:
        """
        Generate a literature review on a topic.
        
        Args:
            topic: Research topic
            sources: List of source documents
            max_sources: Maximum sources to include
            
        Returns:
            Literature review with sections
        """
        logger.info(f"Generating literature review on: {topic}")
        
        # Limit sources
        sources = sources[:max_sources]
        
        # Format sources for prompt
        formatted_sources = self._format_sources(sources)
        
        # Generate review
        prompt = LITERATURE_REVIEW_PROMPT.format(
            topic=topic,
            sources=formatted_sources
        )
        
        try:
            review_text = self.client.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=3000
            )
            
            return {
                "topic": topic,
                "num_sources": len(sources),
                "review": review_text,
                "sources": [s.get("metadata", {}).get("source", "Unknown") for s in sources]
            }
            
        except Exception as e:
            logger.error(f"Error generating review: {e}")
            return {
                "topic": topic,
                "num_sources": 0,
                "review": "Error generating review",
                "sources": []
            }
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for the prompt."""
        
        formatted = []
        
        for i, source in enumerate(sources, 1):
            source_name = source.get("metadata", {}).get("source", f"Source {i}")
            content = source.get("content", "")[:1000]  # Truncate
            
            formatted.append(f"[{i}] {source_name}\n{content}\n")
        
        return "\n---\n".join(formatted)
    
    def export_review(
        self,
        review: Dict[str, Any],
        format: str = "markdown"
    ) -> str:
        """
        Export review to markdown or LaTeX.
        
        Args:
            review: Review data
            format: "markdown" or "latex"
            
        Returns:
            Formatted review
        """
        if format == "markdown":
            return self._export_markdown(review)
        elif format == "latex":
            return self._export_latex(review)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_markdown(self, review: Dict[str, Any]) -> str:
        """Export to Markdown."""
        
        md = f"# Literature Review: {review['topic']}\n\n"
        md += f"*Based on {review['num_sources']} sources*\n\n"
        md += "---\n\n"
        md += review['review'] + "\n\n"
        md += "## References\n\n"
        
        for i, source in enumerate(review['sources'], 1):
            md += f"{i}. {source}\n"
        
        return md
    
    def _export_latex(self, review: Dict[str, Any]) -> str:
        """Export to LaTeX."""
        
        latex = "\\section{Literature Review}\n\n"
        latex += f"\\subsection{{{review['topic']}}}\n\n"
        latex += f"\\textit{{Based on {review['num_sources']} sources}}\n\n"
        latex += review['review'] + "\n\n"
        latex += "\\subsection{References}\n\n"
        latex += "\\begin{enumerate}\n"
        
        for source in review['sources']:
            latex += f"\\item {source}\n"
        
        latex += "\\end{enumerate}\n"
        
        return latex
