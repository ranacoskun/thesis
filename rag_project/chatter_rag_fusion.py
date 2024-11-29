# Chatter.py

import os
import logging

from utils.helpers import setup_logging
from retriever import Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class Chatter:
    """Core chatbot logic integrating retrieval and response generation with RAG Fusion."""
    def __init__(self, log_file='chatter.log') -> None:
        self.data_dir = 'rag_project/data'
        self.log_file = os.path.join(self.data_dir, log_file)
        setup_logging(self.log_file)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

        if not self.openai_api_key:
            logging.error("OpenAI API key is not set.")
            raise ValueError("OpenAI API key is required.")

        self.retriever = Retriever()
        self.llm = ChatOpenAI(model_name='gpt-4', openai_api_key=self.openai_api_key)
        self.prompt_template = self.create_prompt_template()

    def create_prompt_template(self):
        """
        Creates a prompt template for the fused context.
        """
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an AI assistant specialized in legal documents.\n"
                "Use the following context to answer the question.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer in a clear and concise manner."
            )
        )
        return prompt_template

    def chat(self, query):
        """
        Processes a user query and returns a response using RAG Fusion.
        """
        try:
            logging.info(f"Received query: {query}")

            # Retrieve documents
            docs = self.retriever.retrieve(query=query, k=5)
            if not docs:
                logging.warning("No documents retrieved.")
                return "I'm sorry, but I couldn't find any relevant information."

            # Fuse the documents into a single context
            fused_context = "\n\n".join([doc.page_content for doc in docs])

            # Check token length to avoid exceeding model limits
            max_context_length = 2048  # Adjust based on your model's context window
            if len(fused_context) > max_context_length:
                logging.warning("Fused context is too long. Truncating to fit model limits.")
                fused_context = fused_context[:max_context_length]

            # Format the prompt with the fused context
            prompt = self.prompt_template.format(context=fused_context, question=query)

            # Generate response
            response = self.llm(prompt).content

            logging.info(f"Generated response: {response}")
            return response.strip()
        except Exception as e:
            logging.error(f"Failed to generate response: {e}", exc_info=True)
            return "I'm sorry, but I couldn't process your request at this time."

if __name__ == "__main__":
    chatter = Chatter()

    while True:
        query = input("User: ")
        if query.lower() in ['exit', 'quit']:
            break
        response = chatter.chat(query)
        print(f"Assistant: {response}")
