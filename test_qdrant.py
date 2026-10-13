from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_storage")
print("Local Qdrant client initialized successfully!")
