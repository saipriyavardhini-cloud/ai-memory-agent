from database import LocalMemoryAgent
import os
import shutil
import time

def test_memory_agent():
    print("\n--- Running Automated Logging & CRUD Integration Tests ---")
    
    test_path = "./test_qdrant_storage"
    if os.path.exists(test_path):
        shutil.rmtree(test_path, ignore_errors=True)
        
    try:
        agent = LocalMemoryAgent(storage_path=test_path, collection_name="test_collection")
        
        # Test 1: Add a memory
        test_text = "Python and Qdrant make a powerful local vector search stack."
        agent.add_memory(test_text, memory_id=999, category="backend")
        
        # Test 2: Search for the memory
        results = agent.search_memories("What stack is powerful?", limit=1)
        assert len(results) > 0, "Error: Search returned no results!"
        assert results[0]['text'] == test_text, "Error: Retrieved text mismatch!"
        
        # Test 3: List all memories
        all_mems = agent.list_all_memories()
        assert len(all_mems) == 1, "Error: List count mismatch!"
        
        # Test 4: Delete the memory
        agent.delete_memory(999)
        assert len(agent.list_all_memories()) == 0, "Error: Deletion failed!"
        
        print("\n✅ [SUCCESS] All logging and CRUD tests passed perfectly!\n")
        
    finally:
        if 'agent' in locals() and hasattr(agent, 'client'):
            try:
                agent.client.close()
            except Exception:
                pass
            
        time.sleep(0.5)
        if os.path.exists(test_path):
            shutil.rmtree(test_path, ignore_errors=True)

if __name__ == "__main__":
    test_memory_agent()