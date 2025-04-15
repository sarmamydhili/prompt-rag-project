# Standard library imports
import json
import datetime
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional

# Third-party imports
from tqdm import tqdm
from pymongo import MongoClient

# Local imports
import config
from pipeline.pipeline_utils.llm_connections import call_llm_api
from pipeline.pipeline_utils.db_connections import get_mongo_connection, get_chroma_connection, save_to_mongodb, save_to_chroma
from pipeline.pipeline_utils.extract_questions import extract_text_and_flag
from pipeline.pipeline_utils.structure_questions import structure_questions_from_chunk
from pipeline.pipeline_utils.embed_questions import embed_question


# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class ExtractionPipeline:
    def __init__(self, config_path="task_config.json"):
        self.load_config(config_path)
        self.initialize_connections()
        
    def load_config(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        # Database config
        self.db_config = self.config["database"]
        
        # Prompt config
        self.prompt_config = self.config["prompts"]
        
        # PDF config
        self.pdf_config = self.config["pdf"]
        
        # Task config
        self.task_config = self.config["task"]
        # Extract subject from task name
        self.task_config["subject"] = self.task_config["name"].split()[1].lower()
        
        # Validate required configurations
        if not self.task_config["subject"]:
            raise ValueError("Subject not specified in configuration")
        if not self.pdf_config["path"]:
            raise ValueError("PDF path not specified in configuration")

    def initialize_connections(self):
        """Initialize database connections"""
        # Initialize MongoDB
        self.mongo_client = MongoClient(config.MONGO_URI)
        self.mongo_db = self.mongo_client[self.db_config["mongo_adaptive_db_name"]]
        
        # Initialize Chroma
        self.chroma_client, self.chroma_collection = get_chroma_connection()
        
        print("✅ Database connections initialized")

    def load_prompts(self, subject):
        """
        Load both system and user prompts for a given subject.
        Returns a tuple of (system_prompt, user_prompt)
        """
        prompts = {}
        for prompt_type in ["system", "user"]:
            prompt_path = self.prompt_config[f"{prompt_type}_prompt_path"]
            try:
                with open(prompt_path, "r") as f:
                    prompts[prompt_type] = f.read()
                print(f"📄 Loaded {prompt_type} prompt: {prompt_path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        return prompts["system"], prompts["user"]

    def extract(self, pdf_path=None):
        print("🔎 Extracting text and checking for diagrams...")
        pdf_path = pdf_path or self.pdf_config["path"]
        extracted_pages = extract_text_and_flag(pdf_path)
        return extracted_pages

    def structure(self, extracted_pages, system_prompt, user_prompt):
        """
        Structure the extracted pages into questions using the provided prompts.
        
        Args:
            extracted_pages: List of extracted pages with text and diagram flags
            system_prompt: The system prompt to use for LLM
            user_prompt: The user prompt template to use for LLM
        """
        print("🛠️ Structuring text...")
        all_questions = []

        for page in tqdm(extracted_pages, desc="Structuring Pages"):
            text = page["text"]
            diagram_flag = page["diagram_required"]
            
            # Format user prompt with page content
            formatted_user_prompt = user_prompt.format(
                text=text,
                diagram_required=diagram_flag,
                topic=self.task_config["subject"],
                skill="",  # TODO: Add skill from config
                level="Analyzing"  # TODO: Add level from config
            )
            
            response = call_llm_api("openai", system_prompt, formatted_user_prompt)
            questions = structure_questions_from_chunk(response, diagram_flag, "openai")
            all_questions.extend(questions)

        return all_questions

    def embed_and_save(self, structured_questions):
        print("💾 Embedding and saving questions...")

        for q in tqdm(structured_questions, desc="Saving Questions"):
            # Save to MongoDB and get ID
            question_id = save_to_mongodb(
                q,
                self.mongo_db,
                self.db_config["mongo_questions_collection"]
            )

            # Prepare metadata for Chroma
            metadata = {
                "multiple_choices": q.get("multiple_choices", []),
                "correct_answer": q.get("correct_answer", ""),
                "level": q.get("level", "Remembering"),
                "level_num": q.get("level_num", 1),
                "skill": q.get("skill", ""),
                "skill_id": q.get("skill_id", 0),
                "skill_name": q.get("skill_name", ""),
                "subject": q.get("subject", ""),
                "subject_area": q.get("subject_area", ""),
                "subject_area_id": q.get("subject_area_id", 0),
                "subject_id": q.get("subject_id", 0),
                "diagram_required": q.get("diagram_required", False),
                "source": "college_board",
                "timestamp": datetime.datetime.now().isoformat()
            }

            # Save to Chroma
            save_to_chroma(q["question"], question_id, metadata)

    def main(self, pdf_path=None):
        """
        Main pipeline method that orchestrates the entire process:
        1. Extract text from PDF
        2. Structure the text into questions
        3. Save questions to databases
        """
        try:
            print("🚀 Starting pipeline...")
            
            # Step 1: Extract text from PDF
            print("\n📄 Step 1: Extracting text...")
            extracted_pages = self.extract(pdf_path)
            print(f"✅ Extracted {len(extracted_pages)} pages")
            print("\n📝 Extracted Text Samples:")
            for i, page in enumerate(extracted_pages[:2]):  # Show first 2 pages
                print(f"\nPage {i+1}:")
                print(f"Text: {page['text'][:200]}...")  # First 200 chars
                print(f"Diagram Required: {page['diagram_required']}")
            
            # Step 2: Load prompts
            print("\n📝 Step 2: Loading prompts...")
            system_prompt, user_prompt = self.load_prompts(self.task_config["subject"])
            print("✅ Prompts loaded successfully")
            print("\n📝 Prompt Samples:")
            print(f"System Prompt: {system_prompt[:100]}...")
            print(f"User Prompt: {user_prompt[:100]}...")
            
            # Step 3: Structure questions
            print("\n🛠️ Step 3: Structuring questions...")
            structured_questions = self.structure(extracted_pages, system_prompt, user_prompt)
            print(f"✅ Generated {len(structured_questions)} questions")
            print("\n📝 Generated Questions Samples:")
            for i, q in enumerate(structured_questions[:2]):  # Show first 2 questions
                print(f"\nQuestion {i+1}:")
                print(f"Question: {q['question'][:100]}...")
                print(f"Correct Answer: {q['correct_answer']}")
                print(f"Level: {q['level']}")
                print(f"Diagram Required: {q['diagram_required']}")
            
            # Step 4: Save to databases
            print("\n💾 Step 4: Saving questions...")
            self.embed_and_save(structured_questions)
            print("✅ Questions saved successfully")
            print("\n📝 Database Status:")
            print(f"- Saved {len(structured_questions)} questions to MongoDB")
            print(f"- Embedded {len(structured_questions)} questions in Chroma")
            
            print("\n✨ Pipeline completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    pipeline = ExtractionPipeline()
    pipeline.main()