
import asyncio
from llm.ollama_client import OllamaClient
import time

async def test_async_capabilities():
    client = OllamaClient()
    
    print("Verifying setup...")
    client.verify_setup() # This checks for the restored method
    
    print("Testing Async Generation...")
    start = time.time()
    response = await client.agenerate("What is 2+2? Answer in one word.")
    print(f"Response: {response} ({time.time() - start:.2f}s)")
    
    print("\nTesting Async Batch Embedding (5 texts)...")
    texts = ["Hello world", "Machine learning", "Python programming", "Async io", "Vector database"]
    start = time.time()
    embeddings = await client.aembed_batch(texts)
    print(f"Generated {len(embeddings)} embeddings ({time.time() - start:.2f}s)")
    print(f"Dimension: {len(embeddings[0])}")

if __name__ == "__main__":
    try:
        asyncio.run(test_async_capabilities())
        print("\n✅ Async tests passed!")
    except Exception as e:
        print(f"\n❌ Async tests failed: {e}")
