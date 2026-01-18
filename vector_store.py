from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid

# Free local embeddings (fast & good enough)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def setup_vector_store(text_content: str):
    """
    Ingests text, splits it, and stores in a volatile ChromaDB instance.
    """
    # 1. Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = [Document(page_content=text_content)]
    splits = text_splitter.split_documents(docs)

    # 2. Store in Vector DB (Ephemeral/In-memory for this session)
    # Using a unique collection name to avoid collision in a real app
    collection_name = f"session_{uuid.uuid4()}"
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name
    )
    
    return vectorstore

def get_relevant_context(vectorstore, query: str):
    """
    Retrieves the most relevant parts of the resume for the JD.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    # Combine retrieved docs into a single string
    return "\n\n".join([doc.page_content for doc in docs])