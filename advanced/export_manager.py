"""
Export Utilities - Export answers and analyses to various formats
"""

from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from loguru import logger


class ExportManager:
    """Manage exports to different formats."""
    
    def __init__(self):
        """Initialize export manager."""
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)
        logger.info("Initialized export manager")
    
    def export_answer(
        self,
        question: str,
        answer_data: Dict[str, Any],
        format: str = "markdown",
        filename: str = None
    ) -> str:
        """
        Export an answer to a file.
        
        Args:
            question: Original question
            answer_data: Answer data with citations
            format: "markdown" or "latex"
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to exported file
        """
        if format == "markdown":
            content = self._format_answer_markdown(question, answer_data)
            ext = ".md"
        elif format == "latex":
            content = self._format_answer_latex(question, answer_data)
            ext = ".tex"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"answer_{timestamp}{ext}"
        
        # Write file
        filepath = self.export_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Exported answer to {filepath}")
        return str(filepath)
    
    def _format_answer_markdown(
        self,
        question: str,
        answer_data: Dict[str, Any]
    ) -> str:
        """Format answer as Markdown."""
        
        md = f"# Research Question\n\n"
        md += f"**Q**: {question}\n\n"
        md += "---\n\n"
        md += "## Answer\n\n"
        md += answer_data.get("answer", "") + "\n\n"
        md += "---\n\n"
        md += "## Metadata\n\n"
        md += f"- **Confidence**: {answer_data.get('confidence', 0):.0%}\n"
        md += f"- **Sources**: {answer_data.get('num_sources', 0)}\n"
        md += f"- **Citations**: {len(answer_data.get('citations', []))}\n\n"
        
        if answer_data.get("citations"):
            md += "## Citations\n\n"
            for i, citation in enumerate(answer_data["citations"], 1):
                md += f"### [{i}] {citation['source_name']}\n\n"
                md += f"**Section**: {citation.get('section', 'N/A')}\n\n"
                md += f"**Content**:\n```\n{citation.get('content', '')}\n```\n\n"
        
        md += f"\n---\n\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return md
    
    def _format_answer_latex(
        self,
        question: str,
        answer_data: Dict[str, Any]
    ) -> str:
        """Format answer as LaTeX."""
        
        latex = "\\documentclass{article}\n"
        latex += "\\usepackage[utf8]{inputenc}\n"
        latex += "\\usepackage{hyperref}\n\n"
        latex += "\\begin{document}\n\n"
        latex += "\\section{Research Question}\n\n"
        latex += f"\\textbf{{Question}}: {self._escape_latex(question)}\n\n"
        latex += "\\section{Answer}\n\n"
        latex += self._escape_latex(answer_data.get("answer", "")) + "\n\n"
        latex += "\\section{Metadata}\n\n"
        latex += "\\begin{itemize}\n"
        latex += f"\\item \\textbf{{Confidence}}: {answer_data.get('confidence', 0):.0%}\n"
        latex += f"\\item \\textbf{{Sources}}: {answer_data.get('num_sources', 0)}\n"
        latex += f"\\item \\textbf{{Citations}}: {len(answer_data.get('citations', []))}\n"
        latex += "\\end{itemize}\n\n"
        
        if answer_data.get("citations"):
            latex += "\\section{Citations}\n\n"
            for i, citation in enumerate(answer_data["citations"], 1):
                latex += f"\\subsection{{{self._escape_latex(citation['source_name'])}}}\n\n"
                latex += f"\\textbf{{Section}}: {self._escape_latex(citation.get('section', 'N/A'))}\n\n"
                latex += "\\begin{verbatim}\n"
                latex += citation.get('content', '')[:500] + "\n"
                latex += "\\end{verbatim}\n\n"
        
        latex += "\\end{document}\n"
        
        return latex
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = {
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}',
            '\\': '\\textbackslash{}'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def export_bibliography(
        self,
        citations: List[Dict[str, Any]],
        format: str = "bibtex",
        filename: str = None
    ) -> str:
        """
        Export citations as bibliography.
        
        Args:
            citations: List of citations
            format: "bibtex" or "markdown"
            filename: Optional filename
            
        Returns:
            Path to exported file
        """
        if format == "bibtex":
            content = self._format_bibtex(citations)
            ext = ".bib"
        elif format == "markdown":
            content = self._format_bibliography_markdown(citations)
            ext = ".md"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bibliography_{timestamp}{ext}"
        
        filepath = self.export_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Exported bibliography to {filepath}")
        return str(filepath)
    
    def _format_bibtex(self, citations: List[Dict[str, Any]]) -> str:
        """Format citations as BibTeX."""
        
        bibtex = ""
        
        for i, citation in enumerate(citations, 1):
            source = citation.get('source_name', f'source{i}')
            key = source.replace('.', '_').replace(' ', '_')
            
            bibtex += f"@article{{{key},\n"
            bibtex += f"  title={{{source}}},\n"
            bibtex += f"  note={{Section: {citation.get('section', 'N/A')}}}\n"
            bibtex += "}\n\n"
        
        return bibtex
    
    def _format_bibliography_markdown(self, citations: List[Dict[str, Any]]) -> str:
        """Format bibliography as Markdown."""
        
        md = "# Bibliography\n\n"
        
        for i, citation in enumerate(citations, 1):
            md += f"{i}. **{citation.get('source_name', 'Unknown')}**\n"
            md += f"   - Section: {citation.get('section', 'N/A')}\n\n"
        
        return md
