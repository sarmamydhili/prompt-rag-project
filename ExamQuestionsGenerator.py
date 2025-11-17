#!/usr/bin/env python3
"""
ExamQuestionsGenerator.py

A Python program that generates exam questions using LLM models and prompt templates.
This program uses the llm_connections.py module to call various LLM APIs and generates
questions based on the exam_generation_system_prompt.txt and exam_generation_user_prompt.txt files.
"""

import os
import json
import sys
import argparse
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import math
import hashlib
import random

# Add the pipeline directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

from pipeline.pipeline_utils.llm_connections import LLMConnections
from pipeline.pipeline_utils.mongo_operations import MongoOperations
from pipeline.pipeline_utils.db_connections import DBConfig


class ExamQuestionsGenerator:
    """
    A class to generate exam questions using LLM models and prompt templates.
    """
    
    def __init__(self, provider: str = 'openai', model: str = None, temperature: float = 0.3, total_questions: int = 45, top_p: float = 0.95, presence_penalty: float = 0.1, frequency_penalty: float = 0.1, enforce_cross_run_uniqueness: bool = True, similarity_threshold: float = 0.9, max_retries_per_item: int = 3, uniqueness_mongo_collection: str = 'test_questions', hash_field: str = 'hash', embedding_field: str = 'embedding'):
        """
        Initialize the ExamQuestionsGenerator with LLM provider and settings.
        
        Args:
            provider: LLM provider to use (openai, anthropic, gemini, deepseek, grok)
            model: Model name for the provider (optional)
            temperature: Sampling temperature for the LLM
            total_questions: Total number of questions to generate for the exam
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.total_questions = total_questions
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.enforce_cross_run_uniqueness = enforce_cross_run_uniqueness
        # similarity_threshold retained for backward compatibility; not used when cosine disabled
        self.similarity_threshold = similarity_threshold
        self.max_retries_per_item = max_retries_per_item
        self.uniqueness_mongo_collection = uniqueness_mongo_collection
        self.hash_field = hash_field
        self.embedding_field = embedding_field
        self.seen_file_path = "seen_questions.json"
        # Run seed to influence variation
        self.run_seed = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
        # Load local seen hashes
        self._seen_hashes = set()
        try:
            if os.path.exists(self.seen_file_path):
                with open(self.seen_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._seen_hashes = set(h for h in data if isinstance(h, str))
        except Exception:
            self._seen_hashes = set()
        
        # Initialize database configuration
        self._initialize_db_config()
        
        # Initialize LLM connections
        self.llm_connections = LLMConnections({
            'openai_llm_model': model or 'gpt-4o',
            'anthropic_llm_model': model or 'claude-3-5-sonnet-20241022',
            'gemini_llm_model': model or 'gemini-1.5-pro',
            'deepseek_llm_model': model or 'deepseek-chat',
            #'grok_llm_model': model or 'grok-3',
            'grok_llm_model': 'grok-3-mini'

        })
        
        # Initialize MongoDB operations
        self.mongo_ops = MongoOperations()
        
        # Prompt file paths
        self.system_prompt_path = "pipeline/prompts/exam_generation_system_prompt.txt"
        self.user_prompt_path = "pipeline/prompts/exam_generation_user_prompt.txt"
        
        # Validate prompt files exist
        self._validate_prompt_files()
    
    def _initialize_db_config(self):
        """Initialize database configuration with default values"""
        # Create a simple context object with default values
        class SimpleContext:
            def __init__(self):
                self.mongo_server = '127.0.0.1'
                self.mongo_port = '27017'
                self.mongo_db_name = 'adaptive_learning_docs'
                self.mongo_questions_collection = 'test_questions'
                self.mongo_course_framework_collection = 'course_framework'
                self.mongo_output_collection = 'test_questions'
                self.mongo_adaptive_db_name = 'adaptive_learning_docs'
                self.mysql_host = 'localhost'
                self.mysql_database = 'adaptive_learning'
  
        
        # Initialize DBConfig with the context
        context = SimpleContext()
        DBConfig.initialize_from_context(context)
        print("✅ Database configuration initialized")
    
    def _validate_prompt_files(self):
        required_files = [self.system_prompt_path, self.user_prompt_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"❌ Required prompt file not found: {file_path}")
                raise FileNotFoundError(f"Prompt file not found: {file_path}")
        print("✅ All required prompt files found")
    
    @staticmethod
    def _canonicalize_question_text(question: Dict[str, Any]) -> str:
        """
        Build a canonical text representation for hashing/embedding.
        """
        candidates = [
            str(question.get('question') or ''),
            str(question.get('stem') or ''),
            str(question.get('prompt') or ''),
            str(question.get('title') or '')
        ]
        text = " ".join([c.strip() for c in candidates if c]).strip()
        return re.sub(r'\s+', ' ', text)

    @staticmethod
    def _compute_hash(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _dedupe_questions_by_hash(self, subject: str, questions_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicates using SHA-256 hash within this batch and against a local seen set.
        """
        if not questions_list:
            return []
        prior_hashes = self._seen_hashes if self.enforce_cross_run_uniqueness else set()
        seen_hashes = set()
        unique_questions: List[Dict[str, Any]] = []
        for q in questions_list:
            canonical = self._canonicalize_question_text(q)
            q_hash = self._compute_hash(canonical) if canonical else None
            if not q_hash:
                continue
            if q_hash in seen_hashes or q_hash in prior_hashes:
                continue
            q['hash'] = q_hash
            seen_hashes.add(q_hash)
            unique_questions.append(q)
        return unique_questions

    def _load_prompt_template(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    
    def _format_prompt(self, template: str, **kwargs) -> str:
        return template.format(**kwargs)
    
    def get_subject_units_from_mongodb(self, subject: str) -> List[Dict[str, Any]]:
        """
        Query MongoDB course_framework collection for the subject.
        
        Args:
            subject: The subject to query for
            
        Returns:
            List of units with their weightages and topics
        """
        try:
            print(f"🔍 Querying MongoDB for subject: {subject}")
            
            # Query the course_framework collection
            framework_doc = self.mongo_ops.get_course_framework_by_subject(subject)
            
            if not framework_doc:
                print(f"❌ No course framework found for subject: {subject}")
                return []
            
            # Extract units from the framework document
            units = framework_doc.get('units', [])
            
            if not units:
                print(f"❌ No units found in course framework for subject: {subject}")
                return []
            
            print(f"✅ Found {len(units)} units for subject: {subject}")
            return units
            
        except Exception as e:
            print(f"❌ Error querying MongoDB: {e}")
            return []
    
    def calculate_questions_per_unit(self, units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate number of questions per unit based on weightage.
        
        Args:
            units: List of units with weightages
            
        Returns:
            List of units with calculated question counts
        """
        units_with_questions = []
        total_weightage = sum(unit.get('weightage_percent', 0) for unit in units)
        
        if total_weightage == 0:
            print("❌ Total weightage is 0, cannot calculate question distribution")
            return []
        
        print(f"📊 Total weightage: {total_weightage}")
        print(f"🎯 Total questions to distribute: {self.total_questions}")
        
        for unit in units:
            weightage = unit.get('weightage_percent', 0)
            if weightage > 0:
                # Calculate questions based on weightage proportion
                questions = math.ceil((weightage * self.total_questions) / 100)
                
                unit_with_questions = unit.copy()
                unit_with_questions['calculated_questions'] = questions
                units_with_questions.append(unit_with_questions)
                
                print(f"   📚 Unit: {unit.get('unit', 'Unknown')}")
                print(f"      Weightage: {weightage}%")
                print(f"      Questions: {questions}")
        
        return units_with_questions
    
    def _sanitize_llm_response(self, response: str) -> str:
        """
        Clean up the raw LLM response so it can be parsed as JSON.
        Removes markdown code fences and fixes single backslashes used for LaTeX.
        """
        if not response:
            return response

        cleaned = response.strip()
        cleaned = cleaned.strip('`')
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:].strip()
        cleaned = cleaned.strip('`')

        # Escape single backslashes that would break JSON parsing.
        cleaned = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', cleaned)
        cleaned = re.sub(r'(?<!\\)\\infty', r'\\\\infty', cleaned)
        return cleaned

    def generate_questions(self,
                          subject: str,
                          unit_name: str,
                          learning_objectives: List[str],
                          num_questions: int = 1,
                          test_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generate exam questions using the LLM.
        
        Args:
            subject: The subject
            unit_name: The unit name
            learning_objectives: List of learning objectives
            num_questions: Number of questions to generate
            test_type: Optional test type identifier (e.g., "calculator", "no-calculator")
            
        Returns:
            Generated questions as a dictionary or None if generation failed
        """
        try:
            # Load prompt templates
            system_template = self._load_prompt_template(self.system_prompt_path)
            user_template = self._load_prompt_template(self.user_prompt_path)
            
            # Format learning objectives
            learning_objectives_text = "\n".join([f"- {obj}" for obj in learning_objectives])
            learning_objectives_json = json.dumps(learning_objectives)
            
            # Format prompts with parameters
            system_prompt = self._format_prompt(
                system_template,
                subject=subject,
                subject_area=subject,  # Using subject as subject_area
                unit_name=unit_name,
                num_questions=num_questions,
                provider=self.provider,  # Pass the actual provider name
                test_type=test_type or "unspecified",
                learning_objectives=learning_objectives_text,
                learning_objectives_json=learning_objectives_json
            )
            
            user_prompt = self._format_prompt(
                user_template,
                subject=subject,
                subject_area=subject,  # Using subject as subject_area
                unit_name=unit_name,
                num_questions=num_questions,
                provider=self.provider,  # Pass the actual provider name
                test_type=test_type or "unspecified",
                learning_objectives=learning_objectives_text,
                learning_objectives_json=learning_objectives_json
            )

            # Append uniqueness/diversity constraints (no recent prior list)
            constraints = (
                "\n\nConstraints:\n"
                "- Produce unique, non-overlapping questions within this batch.\n"
                "- Avoid paraphrases or minor variants of prior items.\n"
                "- Cover diverse subtopics, skills, and difficulty levels.\n"
                "- Vary formats (MCQ, short answer, reasoning) when appropriate.\n"
                f"- Use a different angle because run_seed={self.run_seed}.\n"
            )
            user_prompt = user_prompt + constraints
            
            print(f"      🔍 Generating {num_questions} questions for unit: {unit_name} (test_type={test_type or 'unspecified'})")
            
            # Call LLM API
            response = self.llm_connections.call_llm_api(
                provider=self.provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                seed=int(self.run_seed.split('_')[-1]) if self.provider == "grok" else None
            )
            
            if response is None:
                print("      ❌ Failed to generate questions - no response from LLM")
                return None
            
            try:
                sanitized_response = self._sanitize_llm_response(response)
                question_data = json.loads(sanitized_response)
                if test_type:
                    questions_list = question_data.get('questions', [])
                    for question in questions_list:
                        if isinstance(question, dict):
                            question['test_type'] = test_type
                    question_data['test_type'] = test_type
                print("      ✅ Questions generated successfully!")
                return question_data
            except json.JSONDecodeError as e:
                print(f"      ❌ Failed to parse JSON response: {e}")
                return None
                
        except Exception as e:
            print(f"      ❌ Error generating questions: {e}")
            return None
    
    def generate_exam_questions(self,
                                subject: str,
                                unit_name: str = None,
                                test_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate complete exam questions for a subject.
        
        Args:
            subject: The subject to generate questions for
            unit_name: Optional specific unit name. If provided, generates questions only for this unit.
                      If None, generates questions for all units based on weightage.
            test_type: Optional test type identifier (e.g., "calculator", "no-calculator")
            
        Returns:
            List of all generated questions
        """
        print(f"🚀 Starting exam generation for subject: {subject} (test_type={test_type or 'unspecified'})")
        
        # Get units from MongoDB
        units = self.get_subject_units_from_mongodb(subject)
        if not units:
            print("❌ No units found, cannot generate questions")
            return []
        
        # Filter units if specific unit name is provided
        if unit_name:
            filtered_units = [unit for unit in units if unit.get('unit', '').lower() == unit_name.lower()]
            if not filtered_units:
                print(f"❌ Unit '{unit_name}' not found in subject '{subject}'")
                print(f"Available units: {[unit.get('unit', 'Unknown') for unit in units]}")
                return []
            units = filtered_units
            print(f"📝 Generating questions for specific unit: {unit_name}")
            print(f"📊 Total questions to generate: {self.total_questions}")
        else:
            print(f"📊 Total questions to generate: {self.total_questions}")
        
        # Calculate questions per unit based on weightage
        units_with_questions = self.calculate_questions_per_unit(units)
        if not units_with_questions:
            print("❌ Failed to calculate question distribution")
            return []
        
        # Generate questions for each unit
        all_questions = []
        
        for unit in units_with_questions:
            current_unit_name = unit.get('unit', 'Unknown')
            num_questions = unit.get('calculated_questions', 0)
            
            if num_questions > 0:
                print(f"\n📝 Generating {num_questions} questions for unit: {current_unit_name}")
                
                # Extract learning objectives for this unit
                unit_learning_objectives = []
                topics = unit.get('topics', [])
                for topic in topics:
                    objectives = topic.get('objectives', [])
                    # Handle both string and dict formats
                    learning_objectives = []
                    for obj in objectives:
                        if isinstance(obj, str):
                            learning_objectives.append(obj)
                        elif isinstance(obj, dict):
                            desc = obj.get('description', '')
                            if desc:
                                learning_objectives.append(desc)
                    unit_learning_objectives.extend(learning_objectives)
                # Shuffle objectives to encourage variety
                random.shuffle(unit_learning_objectives)
                
                # Generate questions for this unit
                questions = self.generate_questions(
                    subject=subject,
                    unit_name=current_unit_name,
                    learning_objectives=unit_learning_objectives,
                    num_questions=num_questions,
                    test_type=test_type
                )
                
                if questions:
                    # Dedupe across runs and within batch; regenerate if shortfall
                    unit_items = list(questions.get('questions', []))
                    unit_items = self._dedupe_questions_by_hash(subject, unit_items)
                    deficit = max(0, num_questions - len(unit_items))
                    retries = 0
                    while deficit > 0 and retries < self.max_retries_per_item:
                        regen = self.generate_questions(
                            subject=subject,
                            unit_name=current_unit_name,
                            learning_objectives=unit_learning_objectives,
                            num_questions=deficit,
                            test_type=test_type
                        )
                        regen_items = list(regen.get('questions', [])) if regen else []
                        regen_items = self._dedupe_questions_by_hash(subject, regen_items)
                        # merge
                        unit_items.extend(regen_items)
                        # trim to desired
                        if len(unit_items) > num_questions:
                            unit_items = unit_items[:num_questions]
                        deficit = max(0, num_questions - len(unit_items))
                        retries += 1
                    # replace questions in set
                    deduped_set = dict(questions)
                    deduped_set['questions'] = unit_items
                    # Update local seen set with committed items
                    for item in unit_items:
                        if isinstance(item, dict) and isinstance(item.get('hash'), str):
                            self._seen_hashes.add(item['hash'])
                    all_questions.append(deduped_set)
                else:
                    print(f"⚠️  Failed to generate questions for unit: {current_unit_name}")
            else:
                print(f"⚠️  Skipping unit {current_unit_name} - no questions allocated")
        
        if all_questions:
            print(f"\n✅ Generated questions for {len(all_questions)} unit(s)")
            # Persist seen set for future runs
            try:
                with open(self.seen_file_path, 'w', encoding='utf-8') as f:
                    json.dump(sorted(self._seen_hashes), f, indent=2)
            except Exception:
                pass
            return all_questions
        else:
            print("❌ Failed to generate any questions")
            return []
    
    def _add_metadata_to_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add metadata fields to each question (matches save_questions_to_mongodb behavior).
        Returns a deep-copied list so callers can write or persist safely.
        """
        enriched_sets: List[Dict[str, Any]] = []
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for i, question_data in enumerate(questions):
            questions_list = question_data.get('questions', [])
            if not questions_list:
                continue

            enriched_set = question_data.copy()
            enriched_set['questions'] = []

            for j, question in enumerate(questions_list):
                enriched_question = dict(question)
                enriched_question['created_at'] = datetime.now().isoformat()
                enriched_question['batch_id'] = batch_id
                enriched_question['question_index'] = j + 1
                enriched_question['set_index'] = i + 1
                enriched_question['question_type'] = "tests"
                if 'test_type' not in enriched_question and question_data.get('test_type'):
                    enriched_question['test_type'] = question_data['test_type']

                enriched_set['questions'].append(enriched_question)

            enriched_sets.append(enriched_set)

        return enriched_sets

    def save_questions_to_file(self, questions: List[Dict[str, Any]], filename: str = None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_exam_questions/exam_questions_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            enriched_questions = self._add_metadata_to_questions(questions)
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(enriched_questions or questions, file, indent=2, ensure_ascii=False)
            print(f"✅ Questions saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving questions to file: {e}")


def save_questions_to_mongodb(questions: List[Dict[str, Any]], mongo_ops: MongoOperations):
    """
    Save questions to MongoDB output collection.
    
    Args:
        questions: List of question data to save
        mongo_ops: MongoDB operations instance
    """
    try:
        print(f"🔍 Saving {len(questions)} question sets to MongoDB")
        
        for i, question_data in enumerate(questions):
            if 'questions' in question_data and question_data['questions']:
                # Extract individual questions from the question set
                individual_questions = question_data['questions']
                
                for j, question in enumerate(individual_questions):
                    # Add metadata to the question
                    question['created_at'] = datetime.now().isoformat()
                    question['batch_id'] = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    question['question_index'] = j + 1
                    question['set_index'] = i + 1
                    question['question_type'] = "tests"
                    if 'test_type' not in question and question_data.get('test_type'):
                        question['test_type'] = question_data['test_type']
                    # Add uniqueness fields
                    try:
                        canonical = ExamQuestionsGenerator._canonicalize_question_text(question)
                        question['hash'] = ExamQuestionsGenerator._compute_hash(canonical)
                    except Exception as _:
                        pass
                    
                    # Save individual question to MongoDB
                    question_id = mongo_ops.save_question(question)
                    print(f"✅ Saved question {j + 1} from set {i + 1} with ID: {question_id}")
            else:
                print(f"⚠️  Invalid question data in set {i + 1}")
        
        print(f"🎉 Successfully saved all questions to MongoDB output collection")
        
    except Exception as e:
        print(f"❌ Error saving questions to MongoDB: {e}")
        raise

def main():
    # Set your parameters here
    subject = "AP Physics 2"  # Change this to your desired subject
    model = "grok-3-latest"  # Change this to your desired model
    temperature = 0.7  # Change this to your desired temperature
    output_file = None  # Change this to your desired output file path (or None for auto-generated)
    provider = "grok"  # Change this to your desired provider
    total_questions = 44  # Change this to your desired total number of questions
    #unit_name = "Advanced Math"  # Change this to a specific unit name (e.g., "Kinematics") or None for all units
    unit_name = None
    test_type = "calculator"  # Change this to your desired test type (e.g., "calculator", "no-calculator")
    try:
        # Initialize the generator
        generator = ExamQuestionsGenerator(
            provider=provider, 
            model=model, 
            temperature=temperature,
            total_questions=total_questions
        )
        
        # Generate exam questions
        questions = generator.generate_exam_questions(
            subject=subject,
            unit_name=unit_name,
            test_type=test_type
        )
        
        if questions:
            # Save to file
            #generator.save_questions_to_file(questions, output_file)
            
            # Save to MongoDB (commented out - uncomment if you want to save to MongoDB as well)
            save_questions_to_mongodb(questions, generator.mongo_ops)
            
            print(f"\n🎉 Successfully generated questions!")
        else:
            print("❌ No questions were generated")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 