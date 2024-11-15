"""Indexer
Contains a class to process PDF files and embed data into a vector store.
After processing, files are moved to an archive.
"""

import os, logging

from utils.helpers import setup_logging, archive_file
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings


class Indexer:
    def __init__(self, log_file='indexer.log'):
        self.data_dir = "rag_project/data"
        self.log_file = os.path.join(self.data_dir, log_file)
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        setup_logging(self.log_file)

        return

    def run(self):
        try:
            files = [f for f in os.listdir(self.data_dir) if f.lower().endswith('.pdf')]
        except FileNotFoundError:
            logging.error(f"Data directory not found: {self.data_dir}")
            return
        except Exception as e:
            logging.error(f"Error accessing data directory {self.data_dir}: {e}", exc_info=True)
            return

        if not files:
            logging.warning(f"No text files found in {self.data_dir}")
            return

        for filename in files:
            self.append_vector_store(filename)
            archive_file(self.data_dir, filename)

    def append_vector_store(self, filename):
        """
        Loads data, creates embeddings, and appends it to a vector store.
        """
        file_path = os.path.join(self.data_dir, filename)
        try:
            logging.info(f"Processing {file_path}...")
            loader = PyPDFLoader(file_path=file_path)
            docs = loader.load()

            if not docs:
                logging.warning(f"No content found in {file_path}")
                return
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(docs)

            vector_store_path = os.path.join(self.data_dir, filename.split('.')[0])
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(vector_store_path)

            logging.info(f"Saved vector store to {vector_store_path}")
        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}", exc_info=True)

        return

if __name__ == "__main__":
    print("Starting indexing...")
    indexer = Indexer()
    indexer.run()
    print("Indexing completed.")
    