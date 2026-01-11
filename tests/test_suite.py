"""
Comprehensive Test Suite for Research Engine
"""

import sys
from pathlib import Path
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from main import ResearchEngine
from advanced.contradiction_detector import ContradictionDetector
from advanced.paper_comparator import PaperComparator
from advanced.literature_review import LiteratureReviewGenerator
from advanced.export_manager import ExportManager

from loguru import logger


class TestSuite:
    """Comprehensive testing for the research engine."""
    
    def __init__(self):
        """Initialize test suite."""
        logger.info("="*60)
        logger.info("Research Engine Test Suite")
        logger.info("="*60)
        
        self.engine = ResearchEngine()
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }
    
    def test_ollama_connectivity(self):
        """Test 1: Ollama connectivity and models."""
        logger.info("\n[TEST 1] Ollama Connectivity")
        
        try:
            verification = self.engine.ollama_client.verify_setup()
            
            if all(verification.values()):
                self.results["passed"].append("Ollama connectivity")
                logger.info("✓ PASSED: Ollama is running and models are available")
                return True
            else:
                self.results["failed"].append("Ollama connectivity")
                logger.error("✗ FAILED: Ollama setup incomplete")
                for key, value in verification.items():
                    logger.error(f"  - {key}: {'✓' if value else '✗'}")
                return False
        except Exception as e:
            self.results["failed"].append(f"Ollama connectivity: {e}")
            logger.error(f"✗ FAILED: {e}")
            return False
    
    def test_document_ingestion(self):
        """Test 2: Document ingestion."""
        logger.info("\n[TEST 2] Document Ingestion")
        
        try:
            # Check for test documents
            test_docs = Path("test_documents")
            if not test_docs.exists() or not list(test_docs.glob("*.md")):
                self.results["warnings"].append("No test documents found")
                logger.warning("⚠ WARNING: No test documents in test_documents/")
                logger.info("  Creating sample documents...")
                return False
            
            # Ingest
            logger.info("Ingesting test documents...")
            self.engine.ingest_documents(str(test_docs))
            
            # Verify
            doc_count = self.engine.vector_store.count()
            if doc_count > 0:
                self.results["passed"].append(f"Document ingestion ({doc_count} chunks)")
                logger.info(f"✓ PASSED: Ingested {doc_count} chunks")
                return True
            else:
                self.results["failed"].append("Document ingestion (no chunks)")
                logger.error("✗ FAILED: No chunks were indexed")
                return False
                
        except Exception as e:
            self.results["failed"].append(f"Document ingestion: {e}")
            logger.error(f"✗ FAILED: {e}")
            return False
    
    def test_basic_query(self):
        """Test 3: Basic query and answer generation."""
        logger.info("\n[TEST 3] Basic Query")
        
        try:
            question = "What is the EM algorithm?"
            logger.info(f"Query: {question}")
            
            result = self.engine.query(question)
            
            # Check answer
            if result.get("answer") and len(result["answer"]) > 50:
                self.results["passed"].append("Basic query")
                logger.info("✓ PASSED: Generated answer")
                logger.info(f"  Answer length: {len(result['answer'])} chars")
                logger.info(f"  Confidence: {result.get('confidence', 0):.0%}")
                logger.info(f"  Citations: {len(result.get('citations', []))}")
                return True
            else:
                self.results["failed"].append("Basic query (short answer)")
                logger.error("✗ FAILED: Answer too short or missing")
                return False
                
        except Exception as e:
            self.results["failed"].append(f"Basic query: {e}")
            logger.error(f"✗ FAILED: {e}")
            return False
    
    def test_citation_accuracy(self):
        """Test 4: Citation accuracy."""
        logger.info("\n[TEST 4] Citation Accuracy")
        
        try:
            question = "Explain neural networks"
            result = self.engine.query(question)
            
            citations = result.get("citations", [])
            
            if len(citations) > 0:
                # Check citation format
                valid_citations = 0
                for citation in citations:
                    if all(k in citation for k in ["source_name", "section", "content"]):
                        valid_citations += 1
                
                if valid_citations == len(citations):
                    self.results["passed"].append(f"Citation accuracy ({len(citations)} citations)")
                    logger.info(f"✓ PASSED: All {len(citations)} citations are valid")
                    return True
                else:
                    self.results["warnings"].append(f"Some citations incomplete ({valid_citations}/{len(citations)})")
                    logger.warning(f"⚠ WARNING: Only {valid_citations}/{len(citations)} citations are complete")
                    return False
            else:
                self.results["warnings"].append("No citations generated")
                logger.warning("⚠ WARNING: No citations in answer")
                return False
                
        except Exception as e:
            self.results["failed"].append(f"Citation accuracy: {e}")
            logger.error(f"✗ FAILED: {e}")
            return False
    
    def test_knowledge_graph(self):
        """Test 5: Knowledge graph construction."""
        logger.info("\n[TEST 5] Knowledge Graph")
        
        try:
            graph = self.engine.knowledge_graph.graph
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            
            if num_nodes > 0:
                self.results["passed"].append(f"Knowledge graph ({num_nodes} nodes, {num_edges} edges)")
                logger.info(f"✓ PASSED: Graph has {num_nodes} nodes and {num_edges} edges")
                
                # Test entity search
                results = self.engine.knowledge_graph.search_entities("algorithm", limit=5)
                logger.info(f"  Found {len(results)} entities matching 'algorithm'")
                
                return True
            else:
                self.results["warnings"].append("Empty knowledge graph")
                logger.warning("⚠ WARNING: Knowledge graph is empty")
                return False
                
        except Exception as e:
            self.results["failed"].append(f"Knowledge graph: {e}")
            logger.error(f"✗ FAILED: {e}")
            return False
    
    def test_advanced_features(self):
        """Test 6: Advanced features."""
        logger.info("\n[TEST 6] Advanced Features")
        
        try:
            # Get some results for testing
            results = self.engine.retriever.retrieve("machine learning", final_top_k=5)
            
            if len(results) < 2:
                self.results["warnings"].append("Not enough results for advanced features")
                logger.warning("⚠ WARNING: Need at least 2 results for advanced features")
                return False
            
            # Test contradiction detection
            logger.info("Testing contradiction detection...")
            detector = ContradictionDetector(self.engine.ollama_client)
            contradictions = detector.find_contradictions(results[:3])
            logger.info(f"  Found {len(contradictions)} contradictions")
            
            # Test paper comparison
            logger.info("Testing paper comparison...")
            comparator = PaperComparator(self.engine.ollama_client)
            comparison = comparator.compare(results[0], results[1], aspects=["approach"])
            logger.info(f"  Comparison generated: {len(comparison.get('comparisons', {}))} aspects")
            
            # Test export
            logger.info("Testing export...")
            exporter = ExportManager()
            answer_data = {"answer": "Test answer", "confidence": 0.8, "num_sources": 2, "citations": []}
            export_path = exporter.export_answer("Test question", answer_data, format="markdown")
            logger.info(f"  Exported to: {export_path}")
            
            self.results["passed"].append("Advanced features")
            logger.info("✓ PASSED: Advanced features working")
            return True
            
        except Exception as e:
            self.results["failed"].append(f"Advanced features: {e}")
            logger.error(f"✗ FAILED: {e}")
            return False
    
    def test_performance(self):
        """Test 7: Performance benchmarks."""
        logger.info("\n[TEST 7] Performance")
        
        try:
            # Test query speed
            start = time.time()
            result = self.engine.query("What is machine learning?")
            query_time = time.time() - start
            
            logger.info(f"  Query time: {query_time:.2f}s")
            
            if query_time < 30:
                self.results["passed"].append(f"Performance (query: {query_time:.2f}s)")
                logger.info("✓ PASSED: Query completed in acceptable time")
                return True
            else:
                self.results["warnings"].append(f"Slow query ({query_time:.2f}s)")
                logger.warning(f"⚠ WARNING: Query took {query_time:.2f}s (>30s)")
                return False
                
        except Exception as e:
            self.results["failed"].append(f"Performance: {e}")
            logger.error(f"✗ FAILED: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests."""
        logger.info("\nStarting comprehensive test suite...\n")
        
        tests = [
            self.test_ollama_connectivity,
            self.test_document_ingestion,
            self.test_basic_query,
            self.test_citation_accuracy,
            self.test_knowledge_graph,
            self.test_advanced_features,
            self.test_performance
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Test crashed: {e}")
                self.results["failed"].append(f"{test.__name__}: crashed")
            
            time.sleep(1)  # Brief pause between tests
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        logger.info(f"\n✓ PASSED: {len(self.results['passed'])}")
        for item in self.results["passed"]:
            logger.info(f"  - {item}")
        
        if self.results["warnings"]:
            logger.info(f"\n⚠ WARNINGS: {len(self.results['warnings'])}")
            for item in self.results["warnings"]:
                logger.warning(f"  - {item}")
        
        if self.results["failed"]:
            logger.info(f"\n✗ FAILED: {len(self.results['failed'])}")
            for item in self.results["failed"]:
                logger.error(f"  - {item}")
        
        total = len(self.results["passed"]) + len(self.results["failed"])
        pass_rate = (len(self.results["passed"]) / total * 100) if total > 0 else 0
        
        logger.info(f"\nOverall: {pass_rate:.0f}% passed ({len(self.results['passed'])}/{total})")
        logger.info("="*60)


if __name__ == "__main__":
    suite = TestSuite()
    suite.run_all_tests()
