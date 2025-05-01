from typing import List, Dict, Optional
from pymongo import MongoClient
from pipeline.pipeline_utils.db_connections import get_mongo_connection, DBConfig

class MongoOperations:
    def __init__(self):
        self.mongo_client, self.mongo_db = get_mongo_connection()
        self.questions_collection = self.mongo_db[DBConfig.MONGO_QUESTIONS_COLLECTION]

    def get_questions_by_skill(self, skill_name: Optional[str] = None, skill: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Get questions from MongoDB filtered by skill_name or skill
        Args:
            skill_name: Optional skill name to filter by
            skill: Optional skill to filter by
            limit: Maximum number of questions to return
        Returns:
            List of questions
        """
        # Build query based on provided parameters
        query = {}
        if skill_name:
            query["skill_name"] = skill_name
        elif skill:
            query["skill"] = skill
            
        # Apply limit only if specified
        if limit is not None:
            questions = self.questions_collection.find(query).limit(limit)
        else:
            questions = self.questions_collection.find(query)
            
        return list(questions)

    def get_questions_by_subject(self, subject: Optional[str] = None) -> List[Dict]:
        """
        Get questions from MongoDB filtered by subject
        Args:
            subject: Optional subject to filter by
        Returns:
            List of questions
        """
        query = {}
        if subject:
            query["subject"] = subject
        return list(self.questions_collection.find(query))

    def save_question(self, question: Dict) -> str:
        """
        Save a question to MongoDB
        Args:
            question: Dictionary containing question data
        Returns:
            MongoDB document ID
        """
        result = self.questions_collection.insert_one(question)
        return str(result.inserted_id)

    def close(self):
        """Close the MongoDB connection"""
        if self.mongo_client:
            self.mongo_client.close() 