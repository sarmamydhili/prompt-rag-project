from typing import List, Dict, Optional
from pymongo import MongoClient
from pipeline.pipeline_utils.db_connections import get_mongo_connection, DBConfig

class MongoOperations:
    def __init__(self):
        """Initialize MongoDB operations with connection from db_connections"""
        self.mongo_client, self.mongo_db = get_mongo_connection()
        self.questions_collection = self.mongo_db[DBConfig.MONGO_QUESTIONS_COLLECTION]
        self.course_framework_collection = self.mongo_db[DBConfig.MONGO_COURSE_FRAMEWORK_COLLECTION]
        self.output_collection = self.mongo_db[DBConfig.MONGO_OUTPUT_COLLECTION]

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
        Save a question to MongoDB test_questions collection
        Args:
            question: Dictionary containing question data
        Returns:
            MongoDB document ID
        """
        result = self.mongo_db['test_questions'].insert_one(question)
        return str(result.inserted_id)

    def get_course_framework_by_subject(self, subject: str) -> Optional[Dict]:
        """
        Get the complete course framework for a specific subject from MongoDB.
        
        Args:
            subject (str): The subject name (e.g., "AP Physics", "AP Calculus BC")
            
        Returns:
            Optional[Dict]: Complete course framework document or None if not found
        """
        try:
            print(f"Fetching course framework for subject: '{subject}'")
            
            # Query for the specific subject
            framework_doc = self.course_framework_collection.find_one({"subject": subject})
            
            if framework_doc:
                print(f"✓ Found course framework for {subject}")
                print(f"  - Units: {len(framework_doc.get('units', []))}")
                
                # Print unit information for debugging
                for unit in framework_doc.get('units', []):
                    unit_name = unit.get('unit', 'Unknown')
                    weightage = unit.get('weightage_percent', 0)
                    topics_count = len(unit.get('topics', []))
                    print(f"    - {unit_name}: {weightage}% weightage, {topics_count} topics")
                
                return framework_doc
            else:
                print(f"✗ No course framework found for subject: '{subject}'")
                
                # List available subjects for debugging
                available_subjects = self.course_framework_collection.distinct("subject")
                print(f"Available subjects: {available_subjects}")
                
                return None
                
        except Exception as e:
            print(f"Error fetching course framework for subject '{subject}': {str(e)}")
            return None

    def inspect_course_framework(self):
        """
        Inspect the structure of documents in the course framework collection.
        This is a diagnostic method to help understand the data structure.
        """
        try:
            # print("\nDEBUG: Inspecting course framework collection...")
            # print(f"Collection name: {self.course_framework_collection.name}")
            
            # Get total document count
            doc_count = self.course_framework_collection.count_documents({})
            # print(f"Total documents: {doc_count}")
            
            if doc_count > 0:
                # Get a sample document
                sample_doc = self.course_framework_collection.find_one()
                # print("\nSample document structure:")
                # print(f"Subject: {sample_doc.get('subject', 'Not found')}")
                # print(f"Units count: {len(sample_doc.get('units', []))}")
                
                if 'units' in sample_doc:
                    # Print first unit structure
                    first_unit = sample_doc['units'][0]
                    # print("\nFirst unit structure:")
                    # print(f"Unit name: {first_unit.get('unit', 'Not found')}")
                    # print(f"Topics count: {len(first_unit.get('topics', []))}")
                    
                    if 'topics' in first_unit:
                        # Print first topic structure
                        first_topic = first_unit['topics'][0]
                        # print("\nFirst topic structure:")
                        # print(f"Topic name: {first_topic.get('topic', 'Not found')}")
                        # print(f"Objectives count: {len(first_topic.get('objectives', []))}")
                        
                        if 'objectives' in first_topic:
                            # Print first objective
                            first_objective = first_topic['objectives'][0]
                            # print("\nFirst objective structure:")
                            # print(f"Description: {first_objective.get('description', 'Not found')}")
            
            # Get list of all unique subjects
            subjects = self.course_framework_collection.distinct("subject")
            # print("\nAvailable subjects:")
            # for subject in subjects:
            #     print(f"- {subject}")
                
        except Exception as e:
            print(f"Error inspecting course framework: {str(e)}")
            # print("DEBUG: Full error details:", exc_info=True)
            
    def get_unit_objectives(self, subject: str, unit: str) -> List[str]:
        """
        Get all objective descriptions for a specific subject and unit.
        
        Args:
            subject (str): The subject name (e.g., "AP Calculus BC")
            unit (str): The unit name (e.g., "Limits and Continuity")
            
        Returns:
            List[str]: List of objective descriptions for the specified unit
        """
        try:
            print(f"\nDEBUG: Fetching unit objectives for subject: '{subject}', unit: '{unit}'")
            print(f"DEBUG: Using collection: {self.course_framework_collection.name}")
            
            # First, try to get a sample document to understand the data structure
            sample_doc = self.course_framework_collection.find_one({"subject": subject})
            if not sample_doc:
                print(f"DEBUG: No document found for subject: {subject}")
                return []
            
            # Check if objectives are stored as strings or objects
            objectives_are_strings = False
            if 'units' in sample_doc and sample_doc['units']:
                for unit_doc in sample_doc['units']:
                    if unit_doc.get('unit') == unit and 'topics' in unit_doc:
                        for topic in unit_doc['topics']:
                            if 'objectives' in topic and topic['objectives']:
                                # Check if first objective is a string or object
                                first_obj = topic['objectives'][0]
                                objectives_are_strings = isinstance(first_obj, str)
                                print(f"DEBUG: Objectives are stored as {'strings' if objectives_are_strings else 'objects'}")
                                break
                        break
            
            # Build aggregation pipeline based on data structure
            if objectives_are_strings:
                # Objectives are stored as strings directly
                pipeline = [
                    {
                        "$match": {
                            "subject": subject
                        }
                    },
                    {
                        "$unwind": "$units"
                    },
                    {
                        "$match": {
                            "units.unit": unit
                        }
                    },
                    {
                        "$unwind": "$units.topics"
                    },
                    {
                        "$unwind": "$units.topics.objectives"
                    },
                    {
                        "$group": {
                            "_id": 0,
                            "descriptions": {
                                "$push": "$units.topics.objectives"  # Push the string directly
                            }
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "descriptions": 1
                        }
                    }
                ]
            else:
                # Objectives are stored as objects with description field
                pipeline = [
                    {
                        "$match": {
                            "subject": subject
                        }
                    },
                    {
                        "$unwind": "$units"
                    },
                    {
                        "$match": {
                            "units.unit": unit
                        }
                    },
                    {
                        "$unwind": "$units.topics"
                    },
                    {
                        "$unwind": "$units.topics.objectives"
                    },
                    {
                        "$group": {
                            "_id": 0,
                            "descriptions": {
                                "$push": "$units.topics.objectives.description"  # Push the description field
                            }
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "descriptions": 1
                        }
                    }
                ]
            
            print(f"DEBUG: Executing aggregation pipeline for {'string' if objectives_are_strings else 'object'} objectives...")
            result = list(self.course_framework_collection.aggregate(pipeline))
            print(f"DEBUG: Pipeline result: {result}")

            # Return the descriptions array or empty array if no results
            descriptions = result[0]['descriptions'] if result else []
            print(f"DEBUG: Returning {len(descriptions)} objectives")
            return descriptions
            
        except Exception as e:
            print(f"Error fetching unit objectives: {str(e)}")
            # print(f"DEBUG: Full error details:", exc_info=True)
            return []

    def close(self):
        """Close the MongoDB connection"""
        if self.mongo_client:
            self.mongo_client.close() 