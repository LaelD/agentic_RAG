from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List



# Load environment variables from .env file
load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",show_progress_bar=False, chunk_size=50 , retry_min_seconds=10
    )
# vectorstore=PineconeVectorStore(
#     index_name="rag-agriculture",
#     embedding=embeddings,
# )

# Load pdf files from docs folder with error handling
def load_pdf_files(data):
    base_dir = Path(__file__).resolve().parent
    data_path = (base_dir / data).resolve()

    try:
        data_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Failed to ensure directory {data_path}: {exc}")
        return []

    loader = DirectoryLoader(
        str(data_path),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    try:
        docs = loader.load()
    except FileNotFoundError:
        print(f"Directory not found: {data_path}")
        return []
    except Exception as exc:
        print(f"Failed to load PDFs from {data_path}: {exc}")
        return []

    if not docs:
        print(f"No PDF files found in {data_path}.")
        return []

    return docs


extract_text = load_pdf_files("docs")

#cleaning the extracted text

def filter_to_minimal_docs(docs:List[Document] )->List[Document]:
    """
    Given a list of Documment objects , return a new list of Document objects
    containing only 'source' in the metadata and 'page_content'
    """
    minimal_docs : list[Document] = []
    for doc in docs:
        src= doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content, 
                metadata={"source": src}
            )
        )
    return minimal_docs

minimal_extract_text = filter_to_minimal_docs(extract_text)

#split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_docs=text_splitter.split_documents(minimal_extract_text)
log_success = f"Successfully split {len(minimal_extract_text)} documents into {len(splitted_docs)} chunks."
print(log_success)

 #splitting
    

print(f'Split into {len(splitted_docs)} chunks of text')

PineconeVectorStore.from_documents(splitted_docs, embeddings, index_name=os.getenv("INDEX_NAME"))

