"""
Text Cleaner - Normalize and clean extracted text
"""

import re
from loguru import logger


class TextCleaner:
    """Clean and normalize extracted text."""
    
    def __init__(self):
        """Initialize text cleaner."""
        pass
    
    def clean(self, text: str, preserve_code: bool = True) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            preserve_code: Whether to preserve code blocks
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        original_length = len(text)
        
        # Normalize unicode
        text = self._normalize_unicode(text)
        
        # Fix common encoding issues
        text = self._fix_encoding(text)
        
        # Remove excessive whitespace
        text = self._normalize_whitespace(text)
        
        # Remove headers/footers (common patterns)
        text = self._remove_headers_footers(text)
        
        # Fix hyphenation
        text = self._fix_hyphenation(text)
        
        logger.debug(f"Cleaned text: {original_length} → {len(text)} chars")
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # Replace common unicode characters
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...', # Ellipsis
            '\xa0': ' ',    # Non-breaking space
            '\u00a0': ' ',  # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Fix common mojibake patterns
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('â€"', '--')
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _remove_headers_footers(self, text: str) -> str:
        """
        Remove common header/footer patterns.
        
        These often appear as:
        - Page numbers
        - Repeated document titles
        - URLs
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip lines that are just page numbers
            if re.match(r'^\d+$', line_stripped):
                continue
            
            # Skip lines that are just URLs
            if re.match(r'^https?://', line_stripped):
                continue
            
            # Skip very short lines at start/end (likely headers/footers)
            # But keep them if they're in the middle of content
            if len(line_stripped) < 3:
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _fix_hyphenation(self, text: str) -> str:
        """
        Fix hyphenation from line breaks.
        
        Example: "exam-\nple" → "example"
        """
        # Pattern: word- followed by newline and word continuation
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text
    
    def extract_code_blocks(self, text: str) -> tuple[str, list[str]]:
        """
        Extract code blocks from text.
        
        Returns:
            Tuple of (text without code, list of code blocks)
        """
        code_blocks = []
        
        # Pattern for markdown code blocks
        pattern = r'```[\w]*\n(.*?)```'
        
        def replace_code(match):
            code = match.group(1)
            code_blocks.append(code)
            return f"[CODE_BLOCK_{len(code_blocks)-1}]"
        
        text_without_code = re.sub(pattern, replace_code, text, flags=re.DOTALL)
        
        return text_without_code, code_blocks
    
    def restore_code_blocks(self, text: str, code_blocks: list[str]) -> str:
        """Restore code blocks to text."""
        for i, code in enumerate(code_blocks):
            text = text.replace(f"[CODE_BLOCK_{i}]", f"```\n{code}```")
        
        return text
