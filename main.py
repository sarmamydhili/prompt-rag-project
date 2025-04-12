from extract.extract_questions import extract_text_from_pdf
from structure.structure_questions import structure_questions
from embed.embed_questions import embed_question
from database.mongo_setup import save_question_to_mongo
from tqdm import tqdm
import uuid
import os

def chunk_text(text, max_length=3000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

if __name__ == "__main__":
    pdf_path = "data/questions.pdf"
    prompt_template = "prompts/structure_prompt.txt"

    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)

    for chunk in tqdm(chunks, desc="Processing Chunks"):
        questions = structure_questions(chunk, prompt_template)

        for question in questions:
            # Save in MongoDB
            question["_id"] = str(uuid.uuid4())
            save_question_to_mongo(question)

            # Save in Chroma
            embed_question(
                question_text=question["question_text"],
                question_id=question["_id"],
                metadata={
                    "subject": question.get("subject", ""),
                    "difficulty": question.get("difficulty", "")
                }
            )
    print("✅ All questions processed, embedded, and saved!")
