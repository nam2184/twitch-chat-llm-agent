import os
import hashlib
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from util import RAGMetrics, Logger, get_hardware, get_root_dir, trace
import time
from fastapi import UploadFile, File

class RAG:
    def __init__(self, logger : Logger = None, metrics : RAGMetrics = None, max_vectors: int = 10000):
        self.metrics = metrics or RAGMetrics()
        self.logger = logger or Logger(name="Store", id=self.id())
        self.max_vectors = max_vectors
        self.vector_store_twitch = self.prepare_retriever_twitch()
    
    def return_vector_store_twitch(self) -> FAISS:
        return self.vector_store_twitch

    def _read_pdf(self, filepath: str) -> list:
        """Extract text from a PDF document and return a list of text chunks.

        Args:
            filepath (str): Path to the PDF document.

        Returns:
            list: A list of text chunks, empty if filepath is None or invalid.
        """
        if not filepath or not os.path.exists(filepath):
            return []

        loader = PyPDFLoader(filepath)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        return [doc.page_content for doc in loader.load_and_split(text_splitter=text_splitter)]
 
    async def _read_pdf_in_memory(self, file: UploadFile = None) -> list:
        """Extract text from a PDF document and return a list of text chunks.

        Args:
            file (UploadFile, optional): In-memory PDF documentself, .

        Returns:
            list: A list of text chunks, empty if file is None.
        """
        if not file:
            return []

        pdf_bytes = await file.read()
        if not pdf_bytes:
            return []

        blob = Blob.from_data(pdf_bytes, path=file.filename)
        loader = PyPDFLoader(blob)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        return [doc.page_content for doc in loader.load_and_split(text_splitter=text_splitter)]
    
    def compute_content_hash(self, chunks: list, embedding_model_name: str) -> str:
        """Compute a hash based on document content and embedding model name.

        Args:
            chunks (list): List of text chunks.
            embedding_model_name (str): Model name.
            
        Returns:
            Hash string.
        """
        if len(chunks) == 0:
            return ""
        content = "".join(chunks) + embedding_model_name
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def get_vector_store(
        self,
        chunks: list,
        embeddings: HuggingFaceEmbeddings,
        cache_dir: str,
        user_id: int = 0
    ) -> FAISS:
        """Retrieve or create a FAISS vector store for a user, even if no chunks are provided.

        Args:
            chunks (list): List of text chunks (can be empty).
            embeddings (HuggingFaceEmbeddings): Embeddings object.
            cache_dir (str): Base directory to save the vector store, separated per user.
            user_id (int): User identifier for separate stores.

        Returns:
            FAISS: Vector store object.
        """
        os.makedirs(cache_dir, exist_ok=True)
        user_cache_dir = os.path.join(cache_dir, str(user_id)) if user_id is not 0 else os.path.join(cache_dir, "default")
        os.makedirs(user_cache_dir, exist_ok=True)
        hash_file = os.path.join(user_cache_dir, "content_hash.txt")
        faiss_index_path = os.path.join(user_cache_dir, "index.faiss")

        # Compute hash for current data (or empty string if no chunks)
        current_hash = self.compute_content_hash(chunks, embeddings.model_name) if chunks else ""

        # Load existing FAISS if available
        if os.path.exists(faiss_index_path):
            vector_store = FAISS.load_local(
                folder_path=user_cache_dir,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )

            # Merge new chunks if present
            if chunks:
                new_store = FAISS.from_texts(
                    chunks,
                    embedding=embeddings,
                    metadatas=[{"timestamp": time.time()} for _ in chunks],
                )
                vector_store.merge_from(new_store)

                # Trim if over capacity
                current_size = len(vector_store.index_to_docstore_id)
                if current_size > self.max_vectors:
                    vector_store = self._trim_index(vector_store=vector_store, max_vectors=self.max_vectors)
                    vector_store.save_local(user_cache_dir)
                    with open(hash_file, "w") as f:
                        f.write(current_hash)
        else:
            # No existing store: create new if chunks exist, otherwise empty FAISS
            if not chunks:
                chunks = ["placeholder text"]  # could also be something like str(time.time())

            vector_store = FAISS.from_texts(
                    chunks,
                    embedding=embeddings,
                    metadatas=[{"timestamp": time.time()} for _ in chunks],
                )
            vector_store.save_local(user_cache_dir)
            with open(hash_file, "w") as f:
                f.write(current_hash)

        return vector_store
    

    def _trim_index(self, vector_store, max_vectors):
        """Trim FAISS index to maintain max vector limit."""
        ids = list(vector_store.index_to_docstore_id.keys())
        if len(ids) <= max_vectors:
            return vector_store  

        to_remove = ids[: len(ids) - max_vectors]
        self.logger.info(f"Trimming {len(to_remove)} old vectors from FAISS cache...")

        for _id in to_remove:
            doc_id = vector_store.index_to_docstore_id.get(_id)
            if doc_id and doc_id in vector_store.docstore._dict:
                del vector_store.docstore._dict[doc_id]
            vector_store.index_to_docstore_id.pop(_id, None)

        # Rebuild FAISS index cleanly
        texts = [d.page_content for d in vector_store.docstore._dict.values()]
        metadatas = [d.metadata for d in vector_store.docstore._dict.values()]
        new_store = FAISS.from_texts(
            texts,
            embedding=vector_store.embedding_function,
            metadatas=metadatas
        )

        return new_store
    
    async def prepare_retriever_file(self, 
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        file: UploadFile = None,
        user_id: int = None
        ) -> FAISS:

        """Create a vector store retriever with the given embedding model.

        Args:
            embedding_model_name (str, optional): Embedding model that maps text to vectors.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2" (lightweight model).
            
            file (UploadFile, optional): File object. Defaults to File initialized object.

        Returns:
            FAISS: A FAISS vector store object.
        """
        with self.metrics.tracer.start_as_current_span("prepare_retriever") as prepare_retriever:
            # Initialize embeddings
            hardware = get_hardware()
            with self.metrics.tracer.start_as_current_span("embeddings", links=[trace.Link(prepare_retriever.get_span_context())]):
                embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    model_kwargs={"device": hardware}
                    )
            # Create text chunks
            with self.metrics.tracer.start_as_current_span("chunks", links=[trace.Link(prepare_retriever.get_span_context())]):
                chunks = await self._read_pdf_in_memory(file)
                
            with self.metrics.tracer.start_as_current_span("vector_store", links=[trace.Link(prepare_retriever.get_span_context())]):
                vector_store = self.get_vector_store(
                    chunks=chunks,
                    embeddings=embeddings,
                    cache_dir=get_root_dir() + "/tiny-rag-llm-agent/vector_store",
                    user_id=user_id
                )
            
            return vector_store
    
    def prepare_retriever_twitch(self, 
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        filepath : str = get_root_dir() + "/tiny-rag-llm-agent/examples/twitch_chat_sample.pdf",
        ) -> FAISS:
        """Create a vector store retriever for twitch responses with the given embedding model.

        Args:
            embedding_model_name (str, optional): Embedding model that maps text to vectors.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2" (lightweight model).
            
            filepath (str): file path to twitch chat data 

        Returns:
            FAISS: A FAISS vector store object.
        """
        with self.metrics.tracer.start_as_current_span("prepare_retriever_twitch") as prepare_retriever_twitch:
            # Initialize embeddings
            hardware = get_hardware()
            with self.metrics.tracer.start_as_current_span("embeddings", links=[trace.Link(prepare_retriever_twitch.get_span_context())]):
                embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    model_kwargs={"device": hardware}
                    )
            # Create text chunks
            with self.metrics.tracer.start_as_current_span("chunks", links=[trace.Link(prepare_retriever_twitch.get_span_context())]):
                chunks = self._read_pdf(filepath=filepath)
                
            with self.metrics.tracer.start_as_current_span("vector_store", links=[trace.Link(prepare_retriever_twitch.get_span_context())]):
                vector_store_twitch = self.get_vector_store(
                    chunks=chunks,
                    embeddings=embeddings,
                    cache_dir=get_root_dir() + "/tiny-rag-llm-agent/vector_store_twitch",
                )
            return vector_store_twitch
        
      

if __name__ == "__main__":
    retriever = RAG().prepare_retriever_file()
    print("Vector store retriever prepared successfully!")