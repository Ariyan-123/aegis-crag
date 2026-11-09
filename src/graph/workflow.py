from langgraph.graph import StateGraph, END
from src.graph.state import AgentGraphState
from src.graph.nodes import AegisNodes
from src.retrieval.hybrid_search import HybridSearchEngine

def decide_to_generate(state: AgentGraphState) -> str:
    """Conditional Edge: Determines whether to generate an answer or refine the query."""
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    if not documents:
        if retry_count >= 1:
            print("--- [DECISION] Max retries reached with no relevant docs. Routing to generate fallback ---")
            return "generate"
        print("--- [DECISION] No relevant documents found. Routing to query transformation ---")
        return "transform_query"
    
    print("--- [DECISION] Relevant documents identified. Routing to generate ---")
    return "generate"

def build_aegis_graph(search_engine: HybridSearchEngine):
    """Compiles the Self-Corrective RAG StateGraph."""
    nodes = AegisNodes(search_engine=search_engine)
    
    workflow = StateGraph(AgentGraphState)
    
    # Register graph nodes
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("transform_query", nodes.transform_query)
    
    # Establish entry point and linear flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    # Establish conditional branching
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        }
    )
    
    # Loop back from transformation to retrieval
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

if __name__ == "__main__":
    from src.ingestion.chunker import load_documents_from_raw, chunk_documents

    # 1. Ingest and index documents
    docs = load_documents_from_raw()
    chunks = chunk_documents(docs)
    engine = HybridSearchEngine()
    engine.index_documents(chunks)

    # 2. Compile self-corrective graph
    app = build_aegis_graph(engine)

    # 3. Test with a technical query
    query = "What happens if block replicas fall below policy for over 180 seconds?"
    inputs = {"question": query, "retry_count": 0}

    print(f"\n================ STARTING GRAPH RUN ================")
    print(f"Initial Question: {query}\n")
    final_state = app.invoke(inputs)

    print(f"\n================ FINAL GENERATED ANSWER ================")
    print(final_state["generation"])