"""pdf_parser.py
"""
import os, re
import pdfplumber
import logging
import argparse


class PDFParser:
    def __init__(self, input_dir, output_dir, log_file='pdf_parser.log'):
        """
        Initializes the PDFParser class.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.log_file = log_file
        self.setup_logging()
        self.create_output_directory()

    def setup_logging(self):
        """
        Configures logging to log messages to a file.
        """
        logging.basicConfig(
            filename=self.log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.DEBUG
        )
        logging.info("Logging setup complete.")

    def create_output_directory(self):
        """
        Creates the output directory if it does not exist.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Created output directory: {self.output_dir}")
        else:
            logging.info(f"Output directory already exists: {self.output_dir}")

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF file.
        """
        text = ''
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'
                        else:
                            logging.warning(f"No text extracted on page {page_number} of {pdf_path}")
                    except Exception as e:
                        logging.error(f"Error extracting text from page {page_number} of {pdf_path}: {e}")
        except FileNotFoundError:
            logging.error(f"File not found: {pdf_path}")
            raise
        except Exception as e:
            logging.error(f"Error opening PDF file {pdf_path}: {e}")
            raise
        return text

    def clean_text(self, text):
        """
        Cleans extracted text by removing unwanted patterns and characters.
        """
        try:
            text = re.sub(r'\n+', '\n', text)
            text = text.strip()
            text = re.sub(r'\n\d+\n', '\n', text)
            text = re.sub(r'\n[^\n]{0,30}\n', '\n', text)
        except Exception as e:
            logging.error(f"Error cleaning text: {e}")
            raise
        return text

    def process_pdf(self, pdf_filename):
        """
        Processes a single PDF file and saves the cleaned text.
        """
        pdf_path = os.path.join(self.input_dir, pdf_filename)
        try:
            logging.info(f"Processing {pdf_path}...")
            raw_text = self.extract_text_from_pdf(pdf_path)
            if not raw_text:
                logging.warning(f"No text extracted from {pdf_path}")
                return
            cleaned_text = self.clean_text(raw_text)
            if cleaned_text:
                output_filename = os.path.splitext(pdf_filename)[0] + '.txt'
                output_path = os.path.join(self.output_dir, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                logging.info(f"Saved cleaned text to {output_path}")
            else:
                logging.warning(f"No cleaned text to save for {pdf_path}")
        except Exception as e:
            logging.error(f"Failed to process {pdf_path}: {e}")

    def process_all_pdfs(self):
        """
        Processes all PDF files in the input directory.
        """
        pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logging.warning(f"No PDF files found in {self.input_dir}")
            return

        for pdf_filename in pdf_files:
            self.process_pdf(pdf_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean and parse PDFs for vector store conversion.')
    parser.add_argument('input_dir', help='The directory containing PDF files.')
    parser.add_argument('output_dir', help='The directory to save cleaned text files.')
    args = parser.parse_args()

    parser = PDFParser(args.input_dir, args.output_dir)
    parser.process_all_pdfs()
