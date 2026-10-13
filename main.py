from database import LocalMemoryAgent

def main():
    print("\n=== Initializing Production-Grade Local AI Memory Agent ===")
    agent = LocalMemoryAgent()
    memory_counter = 1

    while True:
        print("\n--- Memory Manager Menu ---")
        print("1. Add a new memory")
        print("2. Search memories (RAG Context)")
        print("3. List all stored memories")
        print("4. Delete a memory by ID")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            text = input("Enter memory content: ").strip()
            if text:
                category = input("Enter category (e.g., coding, personal, project) [default: general]: ").strip()
                if not category:
                    category = "general"
                agent.add_memory(text, memory_id=memory_counter, category=category)
                memory_counter += 1
            else:
                print("[Error] Memory text cannot be empty.")
                
        elif choice == "2":
            query = input("Enter your search query or question: ").strip()
            if query:
                results = agent.search_memories(query, limit=3, min_score=0.1)
                print(f"\n--- Search Results for: '{query}' ---")
                if not results:
                    print("[Notice] No sufficiently relevant memories found.")
                else:
                    for i, res in enumerate(results, 1):
                        print(f"{i}. [ID: {res['id']}] [Score: {res['score']}] [{res['category']}] {res['text']}")
                        print(f"   Stored at: {res['created_at']}")
                print("-" * 40)
            else:
                print("[Error] Query cannot be empty.")

        elif choice == "3":
            memories = agent.list_all_memories()
            print("\n--- All Stored Memories ---")
            if not memories:
                print("Database is currently empty.")
            else:
                for mem in memories:
                    print(f"ID: {mem['id']} | Category: [{mem['category']}] | Time: {mem['created_at']}")
                    print(f"   -> {mem['text']}")
                    print("-" * 30)
                    
        elif choice == "4":
            try:
                mem_id = int(input("Enter the Memory ID to delete: ").strip())
                agent.delete_memory(mem_id)
            except ValueError:
                print("[Error] Please enter a valid numeric ID.")
                
        elif choice == "5":
            print("Closing database connection. Shutting down memory agent. Goodbye!")
            agent.close()
            break
        else:
            print("[Error] Invalid option. Please choose between 1 and 5.")

if __name__ == "__main__":
    main()