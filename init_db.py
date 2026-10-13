from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(path="./qdrant_storage")

collection_name = "ai_memories"

# Check if collection exists, create if it doesn't
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created successfully!")
else:
    print(f"Collection '{collection_name}' already exists.")
    