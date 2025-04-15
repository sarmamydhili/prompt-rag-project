import pdfplumber
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract text from a PDF file
    Args:
        pdf_path: Path to the PDF file
    Returns:
        Extracted text as a string, or None if extraction fails
    """
    try:
        raw_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        raw_text += page_text + "\n"
                        logger.debug(f"Successfully extracted text from page {page_num}")
                    else:
                        logger.warning(f"No text extracted from page {page_num}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num}: {str(e)}")
                    continue
        
        if not raw_text.strip():
            logger.warning("No text was extracted from the PDF")
            return None
            
        logger.info(f"Successfully extracted text from PDF: {pdf_path}")
        return raw_text.strip()
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return None
