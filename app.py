from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from database import LocalMemoryAgent

# Global agent variable
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the heavy embedding model and Qdrant client safely on startup."""
    global agent
    print("Initializing LocalMemoryAgent inside FastAPI lifespan...")
    agent = LocalMemoryAgent(storage_path="./render_qdrant_storage", collection_name="render_memories")
    yield
    # Cleanup on shutdown
    if agent:
        agent.close()
    print("Application shutdown complete.")

app = FastAPI(
    title="Local Semantic Memory Agent API",
    description="Production-grade local vector memory agent for RAG context retrieval.",
    version="1.0",
    lifespan=lifespan
)

class MemoryCreate(BaseModel):
    id: int
    text: str
    category: str = "general"

class SearchQuery(BaseModel):
    query: str
    limit: int = 3
    min_score: float = 0.15

@app.get("/")
def home():
    return {"status": "Online", "message": "Local Semantic Memory Agent API is running!"}

@app.post("/memories")
def add_memory(item: MemoryCreate):
    if not agent:
        raise HTTPException(status_code=500, detail="Database agent not initialized.")
    try:
        agent.add_memory(memory_text=item.text, memory_id=item.id, category=item.category)
        return {"success": True, "message": f"Stored Memory ID {item.id}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/memories")
def list_memories():
    if not agent:
        raise HTTPException(status_code=500, detail="Database agent not initialized.")
    return {"memories": agent.list_all_memories()}

@app.post("/search")
def search_memories(search: SearchQuery):
    if not agent:
        raise HTTPException(status_code=500, detail="Database agent not initialized.")
    results = agent.search_memories(query_text=search.query, limit=search.limit, min_score=search.min_score)
    return {"query": search.query, "results": results}

@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: int):
    if not agent:
        raise HTTPException(status_code=500, detail="Database agent not initialized.")
    success = agent.delete_memory(memory_id)
    if success:
        return {"success": True, "message": f"Deleted Memory ID {memory_id}"}
    raise HTTPException(status_code=404, detail=f"Memory ID {memory_id} not found.")
