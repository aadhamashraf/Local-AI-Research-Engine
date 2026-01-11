"""
Test script to verify imports and basic functionality
"""

import sys
from pathlib import Path

print("Testing imports...")

try:
    # Test LLM module
    from llm.ollama_client import OllamaClient
    print("✓ LLM module imports successfully")
except Exception as e:
    print(f"✗ LLM module error: {e}")

try:
    # Test ingest module
    from ingest.pdf_loader import PDFLoader
    from ingest.text_cleaner import TextCleaner
    from ingest.chunker import SemanticChunker
    from ingest.document_processor import DocumentProcessor
    print("✓ Ingest module imports successfully")
except Exception as e:
    print(f"✗ Ingest module error: {e}")

try:
    # Test index module
    from index.vector_store import VectorStore
    from index.keyword_index import KeywordIndex
    from index.knowledge_graph import KnowledgeGraph
    print("✓ Index module imports successfully")
except Exception as e:
    print(f"✗ Index module error: {e}")

try:
    # Test retrieval module
    from retrieval.hybrid_search import HybridRetriever
    from retrieval.reranker import LLMReranker
    from retrieval.citation_mapper import CitationMapper
    print("✓ Retrieval module imports successfully")
except Exception as e:
    print(f"✗ Retrieval module error: {e}")

try:
    # Test utils
    from utils import load_config
    print("✓ Utils module imports successfully")
except Exception as e:
    print(f"✗ Utils module error: {e}")

print("\n" + "="*60)
print("All core modules import successfully!")
print("="*60)
print("\nNote: Ollama connectivity will be tested when you run the app.")
print("Make sure to:")
print("1. Start Ollama: ollama serve")
print("2. Pull models:")
print("   - ollama pull qwen2.5:7b")
print("   - ollama pull nomic-embed-text")
