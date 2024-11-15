"""Retriever
Contains the Retriever class to load and combine vector stores from the data directory and make the similarity search.
"""
import os, logging
from utils.helpers import setup_logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


class Retriever:
    """Retriever class to load and combine vector stores from the data directory and make the similarity search."""
    def __init__(self, log_file='retriever.log'):
        self.data_dir = "rag_project/data"
        self.log_file = os.path.join(self.data_dir, log_file)
        setup_logging(self.log_file)
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        self.docsearch = self.load_vector_stores()

    def load_vector_stores(self) -> FAISS:
        """
        Loads all vector stores from the data directory and combines them.
        """
        try:
            vector_store_dirs = []
            for d in os.listdir(self.data_dir):
                dir_path = os.path.join(self.data_dir, d)
                if os.path.isdir(dir_path) and d != 'archive':
                    index_faiss_path = os.path.join(dir_path, 'index.faiss')
                    index_pkl_path = os.path.join(dir_path, 'index.pkl')
                    if os.path.exists(index_faiss_path) and os.path.exists(index_pkl_path):
                        vector_store_dirs.append(dir_path)
                    else:
                        logging.warning(f"No vector store files found in {dir_path}")
                else:
                    logging.debug(f"Skipping non-directory or archive: {dir_path}")

            if not vector_store_dirs:
                logging.error("No vector stores found in data directory.")
                return None

            all_docs = []
            for dir_path in vector_store_dirs:
                logging.info(f"Loading vector store from {dir_path}...")
                docsearch = FAISS.load_local(
                    dir_path,
                    self.embeddings,
                    index_name='index',
                    allow_dangerous_deserialization=True
                )
                docs = list(docsearch.docstore._dict.values())
                all_docs.extend(docs)

            if not all_docs:
                logging.error("No documents found in any vector store.")
                return None

            logging.info("Creating combined vector store...")
            combined_docsearch = FAISS.from_documents(all_docs, self.embeddings)
            logging.info("Combined vector store created successfully.")
            return combined_docsearch
        except Exception as e:
            logging.error(f"Failed to load and combine vector stores: {e}", exc_info=True)
            return None

    def retrieve(self, query, k=5):
        """
        Retrieves relevant documents for a given query.
        """
        if self.docsearch is None:
            logging.error("Vector store is not loaded.")
            return []

        try:
            results = self.docsearch.similarity_search(query, k=k)
            return results
        except Exception as e:
            logging.error(f"Failed to retrieve documents: {e}", exc_info=True)
            return []
