import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DATA_RAW_DIR

def load_documents_from_raw() -> List[Document]:
    """Loads all PDF, Markdown, and TXT files from the data/raw directory."""
    documents = []
    
    if not DATA_RAW_DIR.exists():
        print(f"Warning: Directory {DATA_RAW_DIR} does not exist.")
        return documents

    for file_path in DATA_RAW_DIR.iterdir():
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
        elif file_path.suffix.lower() in [".md", ".txt"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())
            
    print(f"[Ingestion] Loaded {len(documents)} raw document page(s) from {DATA_RAW_DIR}")
    return documents

def chunk_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Splits documents into overlapping chunks and attaches unique chunk IDs.
    Chunk size 500 characters keeps context tight for high retrieval precision.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    
    raw_chunks = splitter.split_documents(documents)
    
    # Attach unique metadata IDs for citations and tracking
    processed_chunks = []
    for idx, chunk in enumerate(raw_chunks):
        source = Path(chunk.metadata.get("source", "doc")).name
        chunk.metadata["chunk_id"] = f"{source}_chunk_{idx}"
        processed_chunks.append(chunk)
        
    print(f"[Ingestion] Created {len(processed_chunks)} semantic chunks ready for indexing.")
    return processed_chunks

if __name__ == "__main__":
    docs = load_documents_from_raw()
    chunks = chunk_documents(docs)
    for c in chunks[:2]:
        print("\n--- SAMPLE CHUNK ---")
        print(f"ID: {c.metadata['chunk_id']}")
        print(c.page_content)