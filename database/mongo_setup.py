from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["prompt_project"]
questions_collection = db["questions"]

def save_question_to_mongo(question):
    questions_collection.insert_one(question)
