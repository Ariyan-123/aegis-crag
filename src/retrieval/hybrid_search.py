import numpy as np
from typing import List, Dict, Any
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import EMBEDDING_MODEL_NAME

class HybridSearchEngine:
    def __init__(self, collection_name: str = "aegis_docs"):
        self.collection_name = collection_name
        print("[Retrieval] Loading embedding model...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.vector_dim = self.encoder.get_sentence_embedding_dimension()
        
        # In-memory vector database
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
        )
        
        self.documents: List[Document] = []
        self.bm25: BM25Okapi = None

    def index_documents(self, documents: List[Document]):
        """Indexes documents into both Qdrant (dense) and BM25 (sparse)."""
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        
        # 1. Build BM25 index
        tokenized_corpus = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Build Qdrant Dense index
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        points = []
        for idx, (doc, emb) in enumerate(zip(documents, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=emb.tolist(),
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                )
            )
        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        print(f"[Retrieval] Successfully indexed {len(documents)} chunks in BM25 & Qdrant.")

    def search_dense(self, query: str, top_k: int = 4) -> List[int]:
        """Returns doc indices ordered by vector similarity."""
        query_vector = self.encoder.encode(query).tolist()
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points
        return [res.id for res in results]

    def search_sparse(self, query: str, top_k: int = 4) -> List[int]:
        """Returns doc indices ordered by BM25 keyword score."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [int(idx) for idx in top_indices if scores[idx] > 0]

    def hybrid_search(self, query: str, top_k: int = 3, rrf_k: int = 60) -> List[Document]:
        """Combines dense and sparse results using Reciprocal Rank Fusion (RRF)."""
        dense_ranks = self.search_dense(query, top_k=top_k * 2)
        sparse_ranks = self.search_sparse(query, top_k=top_k * 2)
        
        rrf_scores: Dict[int, float] = {}
        
        for rank, doc_id in enumerate(dense_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        for rank, doc_id in enumerate(sparse_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        return [self.documents[doc_id] for doc_id in sorted_doc_ids]


if __name__ == "__main__":
    from src.ingestion.chunker import load_documents_from_raw, chunk_documents

    # Load and chunk docs
    raw_docs = load_documents_from_raw()
    chunks = chunk_documents(raw_docs)
    
    # Initialize and index
    engine = HybridSearchEngine()
    engine.index_documents(chunks)
    
    # Run test query
    test_query = "What is the remediation for error code ERR_BLK_403?"
    print(f"\n[Test Query]: {test_query}")
    retrieved_docs = engine.hybrid_search(test_query, top_k=1)
    
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- TOP RETRIEVED RESULT ---")
        print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"Content:\n{doc.page_content}")