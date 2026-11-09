import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.config import GROQ_API_KEY, LLM_MODEL
from src.graph.state import AgentGraphState
from src.retrieval.hybrid_search import HybridSearchEngine

# Pydantic schema for document relevance binary decision
class GradeHallmark(BaseModel):
    binary_score: str = Field(
        description="Relevance score: 'yes' if document is relevant to the question, 'no' otherwise"
    )

class AegisNodes:
    def __init__(self, search_engine: HybridSearchEngine):
        self.search_engine = search_engine
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0.0
        )
        
    def retrieve(self, state: AgentGraphState) -> Dict[str, Any]:
        """Node: Retrieve relevant documents via hybrid search."""
        question = state["question"]
        print(f"\n--- [NODE: RETRIEVE] Fetching documents for: '{question}' ---")
        documents = self.search_engine.hybrid_search(question, top_k=3)
        return {"documents": documents, "retry_count": state.get("retry_count", 0)}

    def grade_documents(self, state: AgentGraphState) -> Dict[str, Any]:
        """Node: Grade retrieved documents for relevance to filter noise."""
        question = state["question"]
        documents = state["documents"]
        print("--- [NODE: GRADE DOCUMENTS] Validating document relevance ---")
        
        structured_llm_grader = self.llm.with_structured_output(GradeHallmark)
        
        system_prompt = (
            "You are an assessment grader evaluating whether a retrieved document "
            "is relevant to a user question. Answer with 'yes' if the document contains "
            "keywords or semantic relevance to answer the question, otherwise 'no'."
        )
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Document:\n\n{document}\n\nQuestion: {question}")
        ])
        
        grader_chain = grade_prompt | structured_llm_grader
        
        filtered_docs = []
        for doc in documents:
            try:
                res = grader_chain.invoke({"document": doc.page_content, "question": question})
                grade = res.binary_score.strip().lower()
            except Exception:
                grade = "yes"  # Fallback to keep doc on parse error
                
            if grade == "yes":
                filtered_docs.append(doc)
                
        print(f"--- [GRADE RESULT] Retained {len(filtered_docs)}/{len(documents)} relevant documents ---")
        return {"documents": filtered_docs}

    def generate(self, state: AgentGraphState) -> Dict[str, Any]:
        """Node: Generate grounded response using validated documents."""
        question = state["question"]
        documents = state["documents"]
        print("--- [NODE: GENERATE] Synthesizing final answer ---")
        
        context = "\n\n".join([f"[{d.metadata.get('chunk_id')}]: {d.page_content}" for d in documents])
        
        system_prompt = (
            "You are AegisRAG, a precise enterprise technical assistant. "
            "Answer the question strictly using the provided context. If the answer cannot "
            "be deduced from the context, state that you do not have enough information."
        )
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
        gen_chain = gen_prompt | self.llm
        response = gen_chain.invoke({"context": context, "question": question})
        return {"generation": response.content}

    def transform_query(self, state: AgentGraphState) -> Dict[str, Any]:
        """Node: Rewrite search query to improve retrieval on retry."""
        question = state["question"]
        retry_count = state.get("retry_count", 0) + 1
        print(f"--- [NODE: TRANSFORM QUERY] Optimizing query (Attempt #{retry_count}) ---")
        
        re_write_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI query optimizer. Rephrase this query to be more effective for vector and keyword search. Output ONLY the rephrased query."),
            ("human", f"Initial question: {question}")
        ])
        
        chain = re_write_prompt | self.llm
        better_query = chain.invoke({}).content.strip()
        print(f"--- [TRANSFORMED] New Query: '{better_query}' ---")
        return {"question": better_query, "retry_count": retry_count}