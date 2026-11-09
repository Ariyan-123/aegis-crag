from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.ingestion.chunker import load_documents_from_raw, chunk_documents
from src.retrieval.hybrid_search import HybridSearchEngine
from src.graph.workflow import build_aegis_graph

# Initialize FastAPI application
app = FastAPI(
    title="AegisRAG API",
    description="Production-grade Self-Corrective Hybrid Retrieval Engine",
    version="1.0.0"
)

# Startup initialization
search_engine = None
graph_app = None

@app.on_event("startup")
def startup_event():
    global search_engine, graph_app
    print("[API Startup] Ingesting documents and building indices...")
    docs = load_documents_from_raw()
    chunks = chunk_documents(docs)
    
    search_engine = HybridSearchEngine()
    search_engine.index_documents(chunks)
    
    print("[API Startup] Compiling LangGraph workflow...")
    graph_app = build_aegis_graph(search_engine)
    print("[API Startup] System ready.")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_sources: List[str]

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AegisRAG"}

@app.post("/query", response_model=QueryResponse)
def query_aegis(req: QueryRequest):
    if not graph_app:
        raise HTTPException(status_code=503, detail="Graph engine is not initialized.")
    
    try:
        inputs = {"question": req.question, "retry_count": 0}
        result = graph_app.invoke(inputs)
        
        sources = [
            doc.metadata.get("chunk_id", "unknown") 
            for doc in result.get("documents", [])
        ]
        
        return QueryResponse(
            question=req.question,
            answer=result.get("generation", "No answer generated."),
            retrieved_sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    