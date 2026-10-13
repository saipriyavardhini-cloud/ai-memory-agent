from database import LocalMemoryAgent
import os
import shutil

def run_retrieval_evaluation():
    print("\n=== Running Local Memory Retrieval Evaluation Suite ===")
    test_path = "./eval_storage"
    if os.path.exists(test_path):
        shutil.rmtree(test_path, ignore_errors=True)
        
    agent = LocalMemoryAgent(storage_path=test_path, collection_name="eval_collection")
    
    # 1. Seed benchmark knowledge base
    benchmark_memories = [
        (1, "Python and Qdrant make a powerful local vector search stack.", "tech"),
        (2, "Sentence-transformers converts text into dense vector embeddings locally.", "ai"),
        (3, "My favorite weekend hobby is hiking in the Rocky Mountains.", "personal"),
        (4, "The project deadline for the backend microservice is Friday.", "work")
    ]
    
    for mem_id, text, cat in benchmark_memories:
        agent.add_memory(text, memory_id=mem_id, category=cat)
        
    # 2. Define test query test cases (Query, Expected Target ID, Min Threshold)
    test_cases = [
        {
            "query": "What database stack am I using with Python?",
            "expected_id": 1,
            "description": "Semantic tech query"
        },
        {
            "query": "How are sentence embeddings generated?",
            "expected_id": 2,
            "description": "Semantic AI query"
        },
        {
            "query": "What do I like to do on weekends?",
            "expected_id": 3,
            "description": "Semantic personal query"
        },
        {
            "query": "completely unrelated query about quantum astrophysics",
            "expected_id": None, # Should be filtered out by threshold
            "description": "Irrelevant noise query"
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    print("\n--- Evaluating Test Cases ---")
    for tc in test_cases:
        results = agent.search_memories(tc["query"], limit=1, min_score=0.20)
        
        if tc["expected_id"] is None:
            success = len(results) == 0
            print(f"Test [{tc['description']}]: Query='{tc['query']}' -> Results returned: {len(results)} (Passed: {success})")
            if success: passed_tests += 1
        else:
            if results and results[0]["id"] == tc["expected_id"]:
                print(f"Test [{tc['description']}]: Query='{tc['query']}' -> Matched Expected ID {tc['expected_id']} with score {results[0]['score']} (Passed)")
                passed_tests += 1
            else:
                top_id = results[0]["id"] if results else "None"
                print(f"Test [{tc['description']}]: Query='{tc['query']}' -> Failed (Got ID {top_id}, Expected {tc['expected_id']})")
                
    accuracy = (passed_tests / total_tests) * 100
    print(f"\nRetrieval Evaluation Complete: {passed_tests}/{total_tests} Passed ({accuracy:.1f}% Accuracy)")
    
    agent.close()
    if os.path.exists(test_path):
        shutil.rmtree(test_path, ignore_errors=True)

if __name__ == "__main__":
    run_retrieval_evaluation()