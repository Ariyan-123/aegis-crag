from typing import List, TypedDict
from langchain_core.documents import Document

class AgentGraphState(TypedDict):
    """
    State tracking schema for AegisRAG.
    
    Attributes:
        question: The user input query.
        generation: The generated answer from the LLM.
        documents: Working list of retrieved documents.
        retry_count: Track fallback iterations for self-correction.
    """
    question: str
    generation: str
    documents: List[Document]
    retry_count: int