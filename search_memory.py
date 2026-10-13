from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_storage")
collection_name = "ai_memories"

# Use a dummy query vector matching the 1536 size
query_vector = [0.1] * 1536

search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=1
).points

print("\n--- Search Results ---")
for hit in search_result:
    print(f"Found Memory: {hit.payload['text']}")
    print(f"Similarity Score: {hit.score}")
    