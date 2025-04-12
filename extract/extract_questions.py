import pdfplumber

def extract_text_from_pdf(pdf_path):
    raw_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw_text += page.extract_text() + "\n"
    return raw_text
