import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_document(file_path: str):
    try:
        # Open the PDF file
        document = fitz.open(file_path)

        # Check if the document has pages
        if document.page_count == 0:
            print(f"Document {file_path} has no pages.")
            return []

        # Extract text from the entire document
        full_text = ""
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            full_text += page.get_text()

        # Clean up the text sample for processing
        cleaned_text = ' '.join(full_text.split())

        # Use LangChain's RecursiveCharacterTextSplitter to chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(cleaned_text)

        return chunks
        print(f"Successfully loaded and chunked document: {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return []