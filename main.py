"""
Main Application - CLI interface for the research engine
"""

import sys
from pathlib import Path
from loguru import logger

from utils import load_config, setup_logging, ensure_directories
from llm.ollama_client import OllamaClient
from llm.answer_generator import AnswerGenerator
from ingest.chunker import SemanticChunker
from ingest.document_processor import DocumentProcessor
from index.vector_store import VectorStore
from index.keyword_index import KeywordIndex
from index.knowledge_graph import KnowledgeGraph
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import LLMReranker
from retrieval.citation_mapper import CitationMapper


class ResearchEngine:
    """Main research engine orchestrator."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize research engine.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        setup_logging(self.config)
        ensure_directories(self.config)
        
        logger.info("=" * 60)
        logger.info("Local AI Research Engine")
        logger.info("=" * 60)
        
        # Initialize Ollama client
        ollama_config = self.config["ollama"]
        self.ollama_client = OllamaClient(
            base_url=ollama_config["base_url"],
            llm_model=ollama_config["llm_model"],
            embedding_model=ollama_config["embedding_model"],
            timeout=ollama_config["timeout"],
            max_retries=ollama_config["max_retries"]
        )
        
        # Verify Ollama setup
        logger.info("Verifying Ollama setup...")
        verification = self.ollama_client.verify_setup()
        
        if not all(verification.values()):
            logger.error("Ollama setup verification failed!")
            logger.error("Please ensure:")
            logger.error("1. Ollama is running")
            logger.error("2. Required models are pulled:")
            logger.error(f"   - ollama pull {ollama_config['llm_model']}")
            logger.error(f"   - ollama pull {ollama_config['embedding_model']}")
            sys.exit(1)
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Ingestion
        ingest_config = self.config["ingestion"]
        self.chunker = SemanticChunker(
            ollama_client=self.ollama_client,
            target_chunk_size=ingest_config["chunk_size"],
            max_chunk_size=ingest_config["max_chunk_size"],
            overlap=ingest_config["chunk_overlap"]
        )
        
        self.document_processor = DocumentProcessor(
            chunker=self.chunker,
            supported_formats=ingest_config["supported_formats"]
        )
        
        # Indexing
        vector_config = self.config["vector_store"]
        self.vector_store = VectorStore(
            persist_directory=vector_config["persist_directory"],
            collection_name=vector_config["collection_name"],
            distance_metric=vector_config["distance_metric"]
        )
        
        keyword_config = self.config["keyword_index"]
        self.keyword_index = KeywordIndex(
            persist_path=keyword_config["persist_path"],
            k1=keyword_config["k1"],
            b=keyword_config["b"]
        )
        
        # Try to load existing index
        self.keyword_index.load()
        
        graph_config = self.config["knowledge_graph"]
        self.knowledge_graph = KnowledgeGraph(
            db_path=graph_config["db_path"],
            ollama_client=self.ollama_client
        )
        
        # Retrieval
        retrieval_config = self.config["retrieval"]
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            keyword_index=self.keyword_index,
            knowledge_graph=self.knowledge_graph,
            ollama_client=self.ollama_client
        )
        
        self.reranker = LLMReranker(
            ollama_client=self.ollama_client
        )
        
        self.citation_mapper = CitationMapper(
            citation_format=self.config["answer_generation"]["citation_format"]
        )
        
        # Answer generation
        answer_config = self.config["answer_generation"]
        self.answer_generator = AnswerGenerator(
            ollama_client=self.ollama_client,
            temperature=answer_config["temperature"],
            max_tokens=answer_config["max_tokens"],
            citation_format=answer_config["citation_format"]
        )
        
        # Advanced Features
        from advanced.paper_comparator import PaperComparator
        from advanced.contradiction_detector import ContradictionDetector
        from advanced.literature_review import LiteratureReviewGenerator
        from advanced.export_manager import ExportManager
        
        self.paper_comparator = PaperComparator(self.ollama_client)
        self.contradiction_detector = ContradictionDetector(self.ollama_client)
        self.lit_reviewer = LiteratureReviewGenerator(self.ollama_client)
        self.exporter = ExportManager()
        
        logger.info("✓ All components initialized successfully")
    
    def ingest_documents(self, directory: str = None):
        """
        Ingest documents from directory.
        
        Args:
            directory: Directory path (defaults to config)
        """
        directory = directory or self.config["paths"]["documents"]
        
        logger.info(f"Ingesting documents from: {directory}")
        
        # Process documents
        results = self.document_processor.process_directory(
            directory=directory,
            recursive=True,
            extract_entities=True
        )
        
        if not results:
            logger.warning("No documents found to ingest")
            return
        
        # Flatten chunks
        all_chunks = []
        for file_path, chunks in results.items():
            all_chunks.extend(chunks)
        
        logger.info(f"Processing {len(all_chunks)} chunks...")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk["content"] for chunk in all_chunks]
        embeddings = self.ollama_client.embed_batch(texts, batch_size=10)
        
        # Add to vector store
        logger.info("Adding to vector store...")
        doc_ids = self.vector_store.add_documents(all_chunks, embeddings)
        
        # Build keyword index
        logger.info("Building keyword index...")
        self.keyword_index.build_index(all_chunks, doc_ids)
        self.keyword_index.save()
        
        # Build knowledge graph
        logger.info("Building knowledge graph...")
        for chunk in all_chunks:
            self.knowledge_graph.extract_and_add_from_chunk(chunk)
        
        self.knowledge_graph.save()
        
        logger.info("✓ Ingestion complete!")
        logger.info(f"  - {len(all_chunks)} chunks indexed")
        logger.info(f"  - {self.knowledge_graph.graph.number_of_nodes()} entities")
        logger.info(f"  - {self.knowledge_graph.graph.number_of_edges()} relationships")
    
    def query(self, question: str) -> dict:
        """
        Answer a question.
        
        Args:
            question: User's question
            
        Returns:
            Answer dictionary
        """
        logger.info(f"\nQuery: {question}")
        
        # Retrieve
        retrieval_config = self.config["retrieval"]
        results = self.retriever.retrieve(
            query=question,
            vector_top_k=retrieval_config["vector_top_k"],
            keyword_top_k=retrieval_config["keyword_top_k"],
            graph_expansion=True,
            final_top_k=retrieval_config["final_top_k"]
        )
        
        if not results:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "citations": [],
                "confidence": 0.0
            }
        
        # Rerank
        reranked = self.reranker.rerank(
            query=question,
            results=results,
            top_k=retrieval_config["rerank_top_k"]
        )
        
        # Register citations
        evidence = self.citation_mapper.register_sources(reranked)
        
        # Generate answer
        answer_data = self.answer_generator.generate_answer(
            question=question,
            evidence=evidence
        )
        
        return answer_data
    
    def interactive_mode(self):
        """Run interactive query loop."""
        logger.info("\n" + "=" * 60)
        logger.info("Interactive Mode - Type 'quit' to exit")
        logger.info("=" * 60 + "\n")
        
        while True:
            try:
                question = input("\n🔍 Question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    logger.info("Goodbye!")
                    break
                
                # Get answer
                result = self.query(question)
                
                # Display
                print("\n" + "=" * 60)
                print("📝 Answer:")
                print("=" * 60)
                print(result["answer"])
                print("\n" + "-" * 60)
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Sources: {result['num_sources']}")
                print(f"Citations: {len(result['citations'])}")
                
                if result["citations"]:
                    print("\n📚 Citations:")
                    for i, citation in enumerate(result["citations"], 1):
                        print(f"  [{i}] {citation['source_name']} §{citation['section']}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local AI Research Engine")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents from the documents directory"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = ResearchEngine(config_path=args.config)
    
    if args.ingest:
        engine.ingest_documents()
    elif args.query:
        result = engine.query(args.query)
        print("\n" + result["answer"])
    else:
        engine.interactive_mode()


if __name__ == "__main__":
    main()
