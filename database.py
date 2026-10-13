import os
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure professional logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LocalMemoryAgent")

class LocalMemoryAgent:
    def __init__(self, storage_path="./qdrant_storage", collection_name="ai_memories"):
        logger.info(f"Initializing local embedding model (all-MiniLM-L6-v2) with storage path: {storage_path}")
        self.storage_path = storage_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = QdrantClient(path=self.storage_path)
        self.collection_name = collection_name
        
        # Ensure collection exists (Dimension 384 for all-MiniLM-L6-v2)
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
                )
                logger.info(f"Created new local vector collection: '{self.collection_name}'")
            else:
                logger.info(f"Connected to existing local vector collection: '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def _get_embedding(self, text: str) -> list:
        """Generates a 384-dimensional dense vector locally."""
        return self.model.encode(text).tolist()

    def add_memory(self, memory_text: str, memory_id: int, category: str = "general"):
        """Embeds and upserts a new memory with metadata and timestamp into Qdrant with edge-case checks."""
        if not memory_text or not memory_text.strip():
            logger.warning("Rejected attempt to add empty or whitespace-only memory.")
            raise ValueError("Memory text cannot be empty or blank.")
            
        if not isinstance(memory_id, int) or memory_id < 0:
            logger.warning(f"Invalid memory_id provided: {memory_id}")
            raise ValueError("Memory ID must be a positive integer.")

        vector = self._get_embedding(memory_text)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=memory_id,
                        vector=vector,
                        payload={
                            "text": memory_text.strip(),
                            "category": category.strip() if category else "general",
                            "created_at": timestamp
                        },
                    )
                ],
            )
            logger.info(f"Stored Memory ID {memory_id} [{category}]: '{memory_text}'")
        except Exception as e:
            logger.error(f"Database error during upsert for ID {memory_id}: {e}")
            raise

    def search_memories(self, query_text: str, limit=3, min_score=0.15) -> list:
        """Performs local cosine similarity search with threshold filtering and validation."""
        if not query_text or not query_text.strip():
            logger.warning("Search query was empty or blank.")
            return []
            
        try:
            query_vector = self._get_embedding(query_text)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            ).points
            
            memories = []
            for hit in results:
                if hit.score >= min_score:
                    memories.append({
                        "id": hit.id,
                        "score": round(hit.score, 4),
                        "text": hit.payload.get("text", ""),
                        "category": hit.payload.get("category", "general"),
                        "created_at": hit.payload.get("created_at", "Unknown")
                    })
            logger.info(f"Searched query '{query_text}' (Threshold: {min_score}) -> Found {len(memories)} relevant matches.")
            return memories
        except Exception as e:
            logger.error(f"Error executing search query: {e}")
            return []

    def list_all_memories(self) -> list:
        """Retrieves all stored memories from the database safely."""
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=100
            )
            
            memories = []
            for record in records:
                memories.append({
                    "id": record.id,
                    "text": record.payload.get("text", ""),
                    "category": record.payload.get("category", "general"),
                    "created_at": record.payload.get("created_at", "Unknown")
                })
            logger.info(f"Retrieved all records -> Total count: {len(memories)}")
            return memories
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            return []

    def delete_memory(self, memory_id: int):
        """Safely deletes a specific memory entry by its ID with validation."""
        try:
            # Check if record exists first
            records, _ = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id]
            )
            if not records:
                logger.warning(f"Attempted to delete non-existent Memory ID {memory_id}.")
                print(f"\n[Warning] Memory ID {memory_id} does not exist in the database.")
                return False

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[memory_id]
                )
            )
            logger.info(f"Successfully deleted Memory ID {memory_id}.")
            print(f"\n[Success] Deleted Memory ID {memory_id}.")
            return True
        except Exception as e:
            logger.error(f"Error deleting Memory ID {memory_id}: {e}")
            print(f"\n[Error] Could not delete memory ID {memory_id}.")
            return False

    def close(self):
        """Safely closes the client connection to release locks."""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
                logger.info("Database client connection closed cleanly.")
            except Exception as e:
                logger.error(f"Error closing database client: {e}")