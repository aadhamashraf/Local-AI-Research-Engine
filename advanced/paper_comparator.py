"""
Paper Comparator - Compare two documents side by side
"""

from typing import Dict, Any, List
from loguru import logger
import json

from llm.ollama_client import OllamaClient
from llm.prompts import PAPER_COMPARISON_PROMPT


class PaperComparator:
    """Compare two papers or documents."""
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize paper comparator.
        
        Args:
            ollama_client: Ollama client instance
        """
        self.client = ollama_client
        logger.info("Initialized paper comparator")
    
    def compare(
        self,
        doc1: Dict[str, Any],
        doc2: Dict[str, Any],
        aspects: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two documents.
        
        Args:
            doc1: First document (with content and metadata)
            doc2: Second document
            aspects: List of aspects to compare (e.g., ["methodology", "results"])
            
        Returns:
            Comparison results
        """
        aspects = aspects or ["methodology", "findings", "approach"]
        
        source1 = doc1.get("metadata", {}).get("source", "Document 1")
        source2 = doc2.get("metadata", {}).get("source", "Document 2")
        
        logger.info(f"Comparing {source1} vs {source2}")
        
        comparisons = {}
        
        for aspect in aspects:
            comparison = self._compare_aspect(
                doc1["content"],
                doc2["content"],
                source1,
                source2,
                aspect
            )
            comparisons[aspect] = comparison
        
        return {
            "source1": source1,
            "source2": source2,
            "comparisons": comparisons,
            "summary": self._generate_summary(comparisons)
        }
    
    def _compare_aspect(
        self,
        content1: str,
        content2: str,
        source1: str,
        source2: str,
        aspect: str
    ) -> str:
        """Compare documents on a specific aspect."""
        
        # Truncate content
        content1 = content1[:2000]
        content2 = content2[:2000]
        
        prompt = PAPER_COMPARISON_PROMPT.format(
            aspect=aspect,
            source1=source1,
            content1=content1,
            source2=source2,
            content2=content2
        )
        
        try:
            response = self.client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error comparing aspect '{aspect}': {e}")
            return f"Error: Could not compare {aspect}"
    
    def _generate_summary(self, comparisons: Dict[str, str]) -> str:
        """Generate an overall summary of comparisons."""
        
        summary_parts = []
        
        for aspect, comparison in comparisons.items():
            # Extract key points (first 200 chars)
            key_point = comparison[:200].split('.')[0] + "."
            summary_parts.append(f"**{aspect.title()}**: {key_point}")
        
        return "\n".join(summary_parts)
    
    def export_comparison(
        self,
        comparison: Dict[str, Any],
        format: str = "markdown"
    ) -> str:
        """
        Export comparison to markdown or LaTeX.
        
        Args:
            comparison: Comparison results
            format: "markdown" or "latex"
            
        Returns:
            Formatted comparison
        """
        if format == "markdown":
            return self._export_markdown(comparison)
        elif format == "latex":
            return self._export_latex(comparison)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_markdown(self, comparison: Dict[str, Any]) -> str:
        """Export to Markdown format."""
        
        md = f"# Comparison: {comparison['source1']} vs {comparison['source2']}\n\n"
        
        md += "## Summary\n\n"
        md += comparison['summary'] + "\n\n"
        
        md += "## Detailed Comparison\n\n"
        
        for aspect, content in comparison['comparisons'].items():
            md += f"### {aspect.title()}\n\n"
            md += content + "\n\n"
        
        return md
    
    def _export_latex(self, comparison: Dict[str, Any]) -> str:
        """Export to LaTeX format."""
        
        latex = "\\section{Comparison}\n\n"
        latex += f"Comparing \\textit{{{comparison['source1']}}} and \\textit{{{comparison['source2']}}}\n\n"
        
        latex += "\\subsection{Summary}\n\n"
        latex += comparison['summary'].replace("**", "\\textbf{").replace("**", "}") + "\n\n"
        
        latex += "\\subsection{Detailed Comparison}\n\n"
        
        for aspect, content in comparison['comparisons'].items():
            latex += f"\\subsubsection{{{aspect.title()}}}\n\n"
            latex += content + "\n\n"
        
        return latex
