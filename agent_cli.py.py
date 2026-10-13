import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

print("Loading local embedding model...", flush=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize local Qdrant client
client_qdrant = QdrantClient(path="./qdrant_storage")
collection_name = "ai_memories"

# Ensure collection exists (Using size=384 for all-MiniLM-L6-v2)
if not client_qdrant.collection_exists(collection_name):
    client_qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created successfully!", flush=True)

def get_embedding(text):
    return model.encode(text).tolist()

def add_memory(memory_text, memory_id):
    vector = get_embedding(memory_text)
    client_qdrant.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=memory_id,
                vector=vector,
                payload={"text": memory_text},
            )
        ],
    )
    print(f"\n[Success] Stored memory ID {memory_id}: '{memory_text}'", flush=True)

def search_memory(query_text, limit=2):
    query_vector = get_embedding(query_text)
    results = client_qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit
    ).points
    
    print(f"\n--- Search results for: '{query_text}' ---", flush=True)
    if not results:
        print("No matching memories found.", flush=True)
    for hit in results:
        print(f"- [Score: {hit.score:.4f}] {hit.payload['text']}", flush=True)
    print("-" * 40, flush=True)

if __name__ == "__main__":
    print("\n=== Local AI Memory Agent Initialized ===")
    memory_counter = 1
    
    while True:
        print("\nOptions:")
        print("1. Add a new memory")
        print("2. Search memories")
        print("3. Exit")
        
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            text = input("Enter the memory text: ").strip()
            if text:
                add_memory(text, memory_id=memory_counter)
                memory_counter += 1
        elif choice == "2":
            query = input("Enter your search query: ").strip()
            if query:
                search_memory(query)
        elif choice == "3":
            print("Exiting memory agent. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")