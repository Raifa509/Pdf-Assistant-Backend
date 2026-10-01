# step 2 - extract text from pdf

import fitz #pymupdf

def extract_text_from_pdf(pdf_path):
    document=fitz.open(pdf_path)
    
    text=""
    for page in document:
        text+=page.get_text()
    
    document.close()
    return text
    