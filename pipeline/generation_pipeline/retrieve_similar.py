# Standard library imports
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
import os
from dotenv import load_dotenv
from pipeline.pipeline_utils.db_connections import get_chroma_connection, DBConfig

# Third-party imports
import chromadb
from chromadb.utils import embedding_functions

# Local imports
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

@dataclass
class SampleQuestion:
    """Data class to represent a sample question with its metadata"""
    question_text: str
    multiple_choices: List[str]
    correct_answer: str
    skill: str
    topic: str

class QuestionRetriever:
    def __init__(self):
        self.client, self.collection = get_chroma_connection()

    def find_similar_questions(self, subject: str, skill: str, topic: str, n_results: int = 3) -> List[Dict]:
        """
        Find similar questions based on subject, skill, and topic
        Args:
            subject: Subject name
            skill: Skill name
            topic: Topic name
            n_results: Number of results to return
        Returns:
            List of similar questions
        """
        if not self.collection:
            print("Error: Chroma collection not initialized")
            return []

        try:
            # Create query embedding
            query_text = f"{subject} {skill} {topic}"
            
            # Search for similar questions
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"subject": subject, "skill": skill, "topic": topic}
            )
            
            # Format results
            questions = []
            for i in range(len(results['ids'][0])):
                question = {
                    'id': results['ids'][0][i],
                    'question_text': results['metadatas'][0][i]['question_text'],
                    'multiple_choices': results['metadatas'][0][i]['multiple_choices'],
                    'correct_answer': results['metadatas'][0][i]['correct_answer'],
                    'subject': results['metadatas'][0][i]['subject'],
                    'skill': results['metadatas'][0][i]['skill'],
                    'topic': results['metadatas'][0][i]['topic'],
                    'metadata': results['metadatas'][0][i].get('metadata', {})
                }
                questions.append(question)
            
            return questions
        except Exception as e:
            print(f"Error finding similar questions: {e}")
            return []

def get_sample_questions(
    subject: str,
    skill: str,
    topic: str,
    n_results: int = 3
) -> List[Dict]:
    """
    Main function to get sample questions in the required format
    Args:
        subject: The subject area
        skill: The skill name
        topic: The topic name
        n_results: Number of similar questions to return (default: 3)
    Returns:
        List of dictionaries containing formatted sample questions
    """
    retriever = QuestionRetriever()
    sample_questions = retriever.find_similar_questions(
        subject=subject,
        skill=skill,
        topic=topic,
        n_results=n_results
    )

    # Convert SampleQuestion objects to dictionaries
    formatted_questions = []
    for question in sample_questions:
        formatted_questions.append({
            "Question": question['question_text'],
            "Multiple Choices": question['multiple_choices'],
            "Correct Answer": question['correct_answer'],
            "Skill": question['skill'],
            "Topic": question['topic'],
            "Metadata": question['metadata']
        })

    return formatted_questions

# Example usage
if __name__ == "__main__":
    sample_questions = get_sample_questions(
        subject="Physics",
        skill="Kinematics",
        topic="Motion in One Dimension"
    )
    
    for question in sample_questions:
        print("\nSample Question:")
        print(f"Question: {question['Question']}")
        print("Options:")
        for choice in question['Multiple Choices']:
            print(f"  {choice}")
        print(f"Correct Answer: {question['Correct Answer']}")
        print(f"Skill: {question['Skill']}")
        print(f"Topic: {question['Topic']}")
