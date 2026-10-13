from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(path="./qdrant_storage")
collection_name = "ai_memories"

# Let's add a test memory with a dummy vector of size 1536 (matching OpenAI embedding dimensions)
# In your full agent, you'll generate this vector using OpenAI embeddings.
dummy_vector = [0.1] * 1536

client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=1,
            vector=dummy_vector,
            payload={"text": "The user is building a local AI memory agent with Qdrant and Python."},
        )
    ],
)

print("Memory stored successfully in local Qdrant!")
