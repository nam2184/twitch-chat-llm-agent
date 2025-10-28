import os
import hashlib
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from util import RAGMetrics, Logger, get_hardware, get_root_dir, get_doc_dir, trace
from constants import SearchKwargs, SearchType

class RAG:
    def __init__(self, logger : Logger = None, metrics : RAGMetrics = None, filepath : str = None):
        self.metrics = metrics or RAGMetrics()
        self.logger = logger or Logger(name="Store", id=self.id())

    def read_pdf(filepath: str) -> list:
        """Extract text from a PDF document and return a list of text chunks.

        Args:
            filepath (str): Path to the pdf document.

        Returns:
            list: A list of text chunks.
        """
        loader = PyPDFLoader(filepath)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        return loader.load_and_split(text_splitter=text_splitter)  

    def compute_content_hash(self, chunks: list, embedding_model_name: str) -> str:
        """Compute a hash based on document content and embedding model name.

        Args:
            chunks (list): List of text chunks.
            embedding_model_name (str): Model name.
            
        Returns:
            Hash string.
        """
        content = "".join(chunks) + embedding_model_name
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def get_vector_store(self, chunks: list, embeddings: HuggingFaceEmbeddings, cache_dir: str) -> FAISS:
        """Retrieves a vector store from a list of text chunks using the given embeddings.

        Args:
            chunks (list): List of text chunks.
            embeddings (HuggingFaceEmbeddings): Embeddings object.
            cache_dir (str): Directory to save the vector store.
            
        Returns:
            FAISS object.
        """
        # Compute content hash
        embedding_model_name = embeddings.model_name
        current_hash = self.compute_content_hash(chunks, embedding_model_name)
        
        # Check if cached index exists and contains a valid hash
        hash_file = os.path.join(cache_dir, "content_hash.txt")
        if os.path.exists(cache_dir) and os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                cached_hash = f.read().strip()
                
            if current_hash == cached_hash:
                return FAISS.load_local(
                    folder_path=cache_dir,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                import shutil
                shutil.rmtree(cache_dir)
                
        # Create a new vector store
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)    
        
        # Save the new index and hash
        os.makedirs(cache_dir, exist_ok=True)
        vector_store.save_local(cache_dir)
        with open(hash_file, "w") as f:
            f.write(current_hash)    
            
        return vector_store

    def prepare_retriever(self, 
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        file_path:str = None
        ) -> VectorStoreRetriever:
        """Create a vector store retriever with the given embedding model.

        Args:
            embedding_model_name (str, optional): Embedding model that maps text to vectors.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2" (lightweight model).
            
            file_path (str, optional): File path to the PDF file. Defaults to None.

        Returns:
            VectorStoreRetriever: A retriever object.
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
                chunks = self.read_pdf(get_doc_dir() if file_path is None else file_path)
                
            with self.metrics.tracer.start_as_current_span("vector_store", links=[trace.Link(prepare_retriever.get_span_context())]):
                vector_store = self.get_vector_store(
                    chunks=chunks,
                    embeddings=embeddings,
                    cache_dir=get_root_dir() + "llm/vector_store"
                )
            
            # Create a retriever
            with self.metrics.tracer.start_as_current_span("retriever", links=[trace.Link(prepare_retriever.get_span_context())]):
                retriever = vector_store.as_retriever(
                    search_type=SearchType.SIMILARITY,
                    search_kwargs={
                        SearchKwargs.K: 2,
                    }
                )
            
            return retriever

if __name__ == "__main__":
    retriever = RAG().prepare_retriever()
    print("Vector store retriever prepared successfully!")