import os
import dotenv

#List: 1 certain type, Dict: 2 certain types (key,value), Any: uncertain type, Optional: None or a certain type 
from typing import List, Any
import logging
from datetime import datetime
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap


from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader,
    DirectoryLoader)

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.chains import create_history_aware_retriever, create_retrieval_chain -- later
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables from .env file
dotenv.load_dotenv()

# default path for vector DB
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    
    def __init__(self, vector_db_path: str = CHROMA_PATH) -> None:
        """
        Initialize the document processor.
        constructors returns None

        Args:
            vector_db_path: Path where the vector database will be stored.
        """
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_db_path = vector_db_path
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.text_splitter = RecursiveCharacterTextSplitter(
            # separators = ["\n\n", "\n", ".", " ", ""], default for RecursiveCharacterTextSplitter
            chunk_size= 1000,
            chunk_overlap=100,
            length_function=lambda text:len(self.tokenizer.encode(text))
            # text-embedding-3-large token limit is 8192, use token to chunk ensure no chunk exceeds this limit
            
        )
        os.makedirs(self.vector_db_path, exist_ok=True)

    def load_documents(self, directory_path: str) -> List[Document]:
        """
        Load documents from a directory using appropriate document loaders.
        
        Args:
            directory_path: Path to the directory containing documents.
            
        Returns:
            List of loaded documents.
            
        Instructions:
        - Use appropriate loaders based on file types (PDF, TXT, DOCX, etc.)
        - Consider using LangChain's DirectoryLoader to load multiple files
        - Handle exceptions for unsupported file types or corrupted files
        - For PDFs, consider using PyPDFLoader or UnstructuredPDFLoader
        - For web pages, implement WebBaseLoader if needed
        """
        # Initialize a list to store loaded documents
        documents = []

        try:

           # Load TXT files
            txt_loader = DirectoryLoader(
                path=directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            logger.info(f"Loaded {len(txt_docs)} TXT documents.")


            if not documents:
                logger.warning("No supported document types found in the directory.")

        except Exception as e:
            logger.error(f"Error loading documents from {directory_path}: {e}", exc_info=True)

        return documents
        
    
    def split_documents(self, documents: List[Document]) -> List:
        """
        Split documents into smaller chunks for effective embedding.
        
        Args:
            documents: List of documents to split.
            
        Returns:
            List of document chunks.
            
        Instructions:
        - Use RecursiveCharacterTextSplitter or other appropriate splitters
        - Choose appropriate chunk size (e.g., 1000 characters) and overlap (e.g., 100 characters)
        - Consider document structure when splitting (try to maintain coherent chunks)
        - Implement metadata preservation during splitting
        """

        # RecursiveCharacterTextSplitter automatically handles metadata preservation
        try:
            # for all documents in documents list - metadata check for existence 
            for doc in documents:
                if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                    doc.metadata = {}

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
            return chunks
        
        except Exception as e:
            logger.error(f"An error occured during document splitting: {e}", exc_info=True)
            return []
    
        
        # Future improvements: try semantic chunking and agentic chunking 
        

    def create_vectorstore(self, document_chunks: List[Document], store_name: str) -> Any:
        """
        Create a vector store from document chunks using embeddings and save it to disk.
        
        Args:
            document_chunks: List of document chunks to embed.
            store_name: Name of the vector store to create.
            
        Returns:
            Vector store containing the document embeddings.
            
        Instructions:
        - Generate embeddings for each chunk using self.embeddings
        - Create a persistent vector store (FAISS, Chroma, etc.) from the embeddings
        - Save the vector store to the specified path with the given name
        - Add metadata to each document for source attribution
        """
        try:
            # Check if the vector store directory exists
            store_path = os.path.join(self.vector_db_path, store_name)
            logger.debug(f"Creating store at path: {store_path}")  
            logger.debug(f"Absolute path: {os.path.abspath(store_path)}")  
        
            os.makedirs(store_path, exist_ok=True)
            logger.debug(f"Directory created/exists: {os.path.exists(store_path)}") 

            # Create a Chroma vector store
            vectorstore = Chroma.from_documents(
                documents=document_chunks,
                embedding=self.embeddings,
                persist_directory=store_path
            )
        
            # Persist the vector store to disk
            vectorstore.persist()
            logger.debug(f"Persisted vector store to: {vectorstore._persist_directory}") 

            # Debug: Log the created files
            created_files = os.listdir(store_path)
            logger.debug(f"Files in store directory: {created_files}")  # Changed from print
            if not created_files:
                logger.warning("No files created in vector store directory!")

            logger.info(f"Vector store '{store_name}' created successfully at {store_path}")
            return vectorstore
        
        except Exception as e:
            logger.error(f"Error creating vector store: {e}", exc_info=True)
            return None
        
    def process_directory(self, directory_path: str, store_name: str) -> str:
        """
        Complete pipeline to ingest documents from a directory and create a vector store.
        
        Args:
            directory_path: Path to directory containing source documents.
            store_name: Name for the created vector store.
            
        Returns:
            Path to the created vector store.
            
        Instructions:
        - Load documents from the directory
        - Split documents into chunks
        - Create and persist vector store from chunks
        - Return the path to where the vector store is saved
        """
        logger.info(f"Starting document processing pipeline for {directory_path}")
        documents = self.load_documents(directory_path)
        logger.debug(f"Loaded {len(documents)} source documents")
    
        chunks = self.split_documents(documents)
        logger.debug(f"Split into {len(chunks)} chunks")
    
        vectorstore = self.create_vectorstore(chunks, store_name)
    
        full_path = os.path.join(self.vector_db_path, store_name)
        logger.info(f"Processing complete. Vector store available at {full_path}")
        return full_path
    
    def list_available_vectorstores(self) -> List[str]:
        """
        List all available vector stores in the vector_db_path.
        
        Returns:
            List of available vector store names.
            
        Instructions:
        - Check the vector_db_path directory for subdirectories or files representing vector stores
        - Return the names of available vector stores that can be loaded
        """
        try:
            if not os.path.exists(self.vector_db_path):
                logger.warning(f"Vector DB path '{self.vector_db_path}' does not exist.")
                return []
            
            # Print the contents of the vector DB path
            logger.debug(f"Listing vector stores in: {self.vector_db_path}")

            # Only list directories 
            store_names = [
                name for name in os.listdir(self.vector_db_path)
                if os.path.isdir(os.path.join(self.vector_db_path, name))
            ]

            logger.info(f"Found vector stores: {store_names}") 
            
            logger.info(f"Found {len(store_names)} vector store(s).")
            return store_names

        except Exception as e:
            logger.error(f"Error listing vector stores: {e}", exc_info=True)
            return []
    
    # Need to check which vector store this method uses (Additional feature - optional)
    def visualize_query_projection(self, store_name: str, query: str, n_results: int = 5) -> None:
        """
        Visualize document embeddings, a query embedding, and the closest retrieved documents using UMAP.

        Args:
            store_name: Name of the vector store.
            query: User query string to embed and visualize.
            n_results: Number of top-matching documents to retrieve and visualize.
        """
        try:
            # Load the persisted vector store
            store_path = os.path.join(self.vector_db_path, store_name)
            vectorstore = Chroma(
                persist_directory=store_path,
                embedding_function=self.embeddings
            )

            # Retrieve all stored embeddings from Chroma
            all_docs = vectorstore.get(include=["embeddings"])
            doc_embeddings = np.array(all_docs["embeddings"])

            if doc_embeddings.size == 0:
                logger.warning("No embeddings found in the vector store.")
                return

            # Fit UMAP on document embeddings with transform_mode=True so we can later project new points (e.g., query)
            umap_model = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
                transform_mode='embedding'  
            )
            reduced_all = umap_model.fit_transform(doc_embeddings)

            # Generate embedding for the user query
            query_embedding = self.embeddings.embed_query(query)

            # Transform query into same 2D UMAP space
            query_2d = umap_model.transform([query_embedding])

            # Retrieve top-n documents relevant to the query
            retriever = vectorstore.as_retriever(search_kwargs={"k": n_results})
            results = retriever.get_relevant_documents(query)

            # Recompute the embeddings of the retrieved documents
            retrieved_texts = [doc.page_content for doc in results]
            retrieved_embeddings = self.embeddings.embed_documents(retrieved_texts)
            retrieved_2d = umap_model.transform(retrieved_embeddings)

            # --- Visualization ---
            plt.figure(figsize=(10, 8))
            plt.scatter(reduced_all[:, 0], reduced_all[:, 1], alpha=0.5, label="All documents", color="gray")
            plt.scatter(retrieved_2d[:, 0], retrieved_2d[:, 1], edgecolors="green", facecolors="none",
                        s=120, label="Retrieved documents", linewidths=2)
            plt.scatter(query_2d[0][0], query_2d[0][1], color="red", marker="X", s=200, label="Query")
            plt.title(f"UMAP Projection for Query: \"{query}\"")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            plt.grid(True)
            plt.legend()

            # Save plot
            output_dir = "embedding_visualization"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(output_dir, f"query_projection_{timestamp}.png")
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()

            # Log and print
            print(f"Visualization saved to: {filepath}")
            for i, doc in enumerate(results):
                print(f"\n[{i + 1}] Retrieved Document:\n{doc.page_content[:500]}...\n")  

        except Exception as e:
            logger.error(f"Error during embedding visualization: {e}", exc_info=True)
