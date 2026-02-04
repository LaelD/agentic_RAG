import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


class DocumentIngestor:
    """Handles document loading, processing, and ingestion to vector store."""
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the document ingestor.
        
        Args:
            index_name: Pinecone index name (default: from env)
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.index_name = index_name or os.getenv("INDEX_NAME")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            show_progress_bar=False,
            chunk_size=50,
            retry_min_seconds=10
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def load_pdf_files(self, data_folder: str = "docs") -> List[Document]:
        """
        Load PDF files from the specified folder.
        
        Args:
            data_folder: Folder name relative to this script
            
        Returns:
            List of loaded documents
        """
        base_dir = Path(__file__).resolve().parent
        data_path = (base_dir / data_folder).resolve()

        try:
            data_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"❌ Failed to ensure directory {data_path}: {exc}")
            return []

        loader = DirectoryLoader(
            str(data_path),
            glob="*.pdf",
            loader_cls=PyPDFLoader,
        )

        try:
            docs = loader.load()
        except FileNotFoundError:
            print(f"❌ Directory not found: {data_path}")
            return []
        except Exception as exc:
            print(f"❌ Failed to load PDFs from {data_path}: {exc}")
            return []

        if not docs:
            print(f"⚠️ No PDF files found in {data_path}")
            return []

        print(f"✅ Loaded {len(docs)} documents from {data_path}")
        return docs
    
    def clean_documents(self, docs: List[Document]) -> List[Document]:
        """
        Clean documents by keeping only 'source' in metadata.
        
        Args:
            docs: List of documents to clean
            
        Returns:
            List of cleaned documents
        """
        cleaned = []
        for doc in docs:
            src = doc.metadata.get("source")
            cleaned.append(
                Document(
                    page_content=doc.page_content,
                    metadata={"source": src}
                )
            )
        return cleaned
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            docs: List of documents to split
            
        Returns:
            List of document chunks
        """
        splits = self.text_splitter.split_documents(docs)
        print(f"✅ Split {len(docs)} documents into {len(splits)} chunks")
        return splits
    
    def store_in_vectordb(self, docs: List[Document]) -> None:
        """
        Store documents in Pinecone vector database.
        
        Args:
            docs: List of document chunks to store
        """
        PineconeVectorStore.from_documents(
            docs,
            self.embeddings,
            index_name=self.index_name
        )
        print(f"✅ Stored {len(docs)} chunks in Pinecone index '{self.index_name}'")
    
    def ingest(self, docs_folder: str = "docs") -> None:
        """
        Complete ingestion pipeline: load, clean, split, and store documents.
        
        Args:
            docs_folder: Folder containing PDF documents
        """
        print("📂 Loading documents...")
        docs = self.load_pdf_files(docs_folder)
        
        if not docs:
            print("❌ No documents to process")
            return
        
        print("🧹 Cleaning documents...")
        cleaned_docs = self.clean_documents(docs)
        
        print(f"✂️ Splitting documents (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        splits = self.split_documents(cleaned_docs)
        
        print("🔄 Embedding and storing in Pinecone...")
        self.store_in_vectordb(splits)
    
        print("✅ Ingestion complete!")


def ingest_documents(docs_folder: str = "docs", chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Convenience function for quick ingestion.
    
    Args:
        docs_folder: Folder containing PDF documents
        chunk_size: Maximum size of text chunks
        chunk_overlap: Overlap between chunks
    """
    ingestor = DocumentIngestor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ingestor.ingest(docs_folder)


if __name__ == "__main__":
    # דוגמת שימוש : עם ברירת מחדל
    ingest_documents()
    

