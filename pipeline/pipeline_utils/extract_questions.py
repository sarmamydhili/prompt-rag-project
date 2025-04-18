import pdfplumber
import logging
from typing import Optional, List, Dict
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_answer_key_alignment(pdf_path: str) -> Dict[int, Dict]:
    """
    Extract answer key and alignment information from the PDF
    Args:
        pdf_path: Path to the PDF file
    Returns:
        Dictionary mapping question numbers to their alignment info
    """
    try:
        alignment_info = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and ("Answer Key" in text or "Question Alignment" in text):
                    # Extract table from the page
                    tables = page.extract_tables()
                    for table in tables:
                        # Skip header row
                        for row in table[1:]:  # Skip header row
                            try:
                                if len(row) >= 5:  # Ensure row has all required columns
                                    question_num = int(row[0])
                                    alignment_info[question_num] = {
                                        "answer": row[1].strip(),
                                        "skill": row[2].strip(),
                                        "learning_objective": row[3].strip(),
                                        "unit": int(row[4].strip())
                                    }
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Error processing alignment row: {row}, Error: {e}")
                                continue
        
        logger.info(f"Extracted alignment info for {len(alignment_info)} questions")
        return alignment_info
    except Exception as e:
        logger.error(f"Error extracting answer key alignment: {str(e)}")
        return {}

def extract_text_and_flag(pdf_path: str, alignment_path: str = None) -> List[Dict[str, any]]:
    """
    Extract text from a PDF file and flag pages that might contain diagrams
    Args:
        pdf_path: Path to the PDF file
        alignment_path: Path to the alignment PDF file (optional)
    Returns:
        List of dictionaries containing page text and diagram flag
    """
    try:
        pages = []
        # First extract answer key alignment if alignment path is provided
        answer_key_alignment = {}
        if alignment_path:
            answer_key_alignment = extract_answer_key_alignment(alignment_path)
            logger.info(f"Loaded alignment data for {len(answer_key_alignment)} questions")
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text
                    page_text = page.extract_text()
                    
                    # Skip if this is an alignment page
                    if page_text and ("Answer Key" in page_text or "Question Alignment" in page_text):
                        continue
                    
                    # Check for potential diagrams
                    diagram_flag = False
                    if page_text:
                        # Look for common diagram indicators
                        diagram_indicators = [
                            r'figure\s+\d+',  # Figure X
                            r'fig\.\s+\d+',   # Fig. X
                            r'diagram\s+\d+', # Diagram X
                            r'chart\s+\d+',   # Chart X
                            r'graph\s+\d+',   # Graph X
                            r'image\s+\d+',   # Image X
                        ]
                        
                        for pattern in diagram_indicators:
                            if re.search(pattern, page_text, re.IGNORECASE):
                                diagram_flag = True
                                break
                                
                        # Extract question number if present
                        question_match = re.search(r'(?:Question|Q)\s*(\d+)', page_text, re.IGNORECASE)
                        question_number = int(question_match.group(1)) if question_match else None
                        
                        # Get alignment info if question number is found
                        alignment_info = answer_key_alignment.get(question_number, {}) if question_number else {}
                    
                    pages.append({
                        "page_number": page_num,
                        "text": page_text if page_text else "",
                        "diagram_required": diagram_flag,
                        "question_number": question_number if 'question_number' in locals() else None,
                        "answer": alignment_info.get("answer", ""),
                        "skill": alignment_info.get("skill", ""),
                        "learning_objective": alignment_info.get("learning_objective", ""),
                        "unit": alignment_info.get("unit", None)
                    })
                    
                    logger.debug(f"Processed page {page_num}")
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {str(e)}")
                    continue
        
        if not pages:
            logger.warning("No pages were processed from the PDF")
            return []
            
        logger.info(f"Successfully processed {len(pages)} pages from PDF: {pdf_path}")
        return pages
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return []

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
