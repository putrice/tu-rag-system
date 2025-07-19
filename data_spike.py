# data_spike.py
import fitz  # PyMuPDF
import os

def test_pdf_parsing(file_path, num_pages_to_check=5):
    """
    Opens a PDF, extracts text from the first few pages, and prints a sample.

    Args:
        file_path (str): The path to the PDF file.
        num_pages_to_check (int): The number of initial pages to sample text from.

    Returns:
        bool: True if text extraction was successful, False otherwise.
    """
    print("-" * 80)
    print(f"Testing file: {os.path.basename(file_path)}")
    print("-" * 80)

    try:
        # Open the PDF file
        document = fitz.open(file_path)

        # Check if the document has enough pages
        if document.page_count == 0:
            print("Result: FAILED - Document has no pages.")
            return False

        print(f"Document has {document.page_count} pages. Checking the first {num_pages_to_check}.")

        full_text_sample = ""
        # Extract text from the first few pages
        for page_num in range(min(num_pages_to_check, document.page_count)):
            page = document.load_page(page_num)
            full_text_sample += page.get_text()

        # Clean up the text sample for printing
        # Replace multiple newlines with a single one for readability
        cleaned_sample = ' '.join(full_text_sample.split())

        if not cleaned_sample.strip():
            print("Result: FAILED - Extracted text is empty. The PDF might be image-based.")
            return False

        # Print a snippet of the extracted text
        print("\n--- Sample of Extracted Text (first 500 chars) ---")
        print(cleaned_sample[:500])
        print("...\n")

        print("Result: SUCCESS - Text extraction looks clean.")
        return True

    except Exception as e:
        print(f"Result: FAILED - An error occurred: {e}")
        return False

if __name__ == "__main__":
    # Directory containing the 10-K reports
    data_directory = "data"

    # Get all PDF files from the data directory
    candidate_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith(".pdf")]

    if not candidate_files:
        print("No PDF files found in the 'data' directory. Please add your 10-K reports.")
    else:
        print("Starting data spike to test PDF parsing quality...\n")
        for file in candidate_files:
            test_pdf_parsing(file)