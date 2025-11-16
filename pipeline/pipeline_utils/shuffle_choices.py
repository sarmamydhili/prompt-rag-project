import random
from pymongo import MongoClient

def get_mongodb_connection(connection_string):
    """Establish MongoDB connection based on connection string"""
    try:
        client = MongoClient(connection_string)
        # Test the connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB")
        return client
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None

def shuffle_doc(doc):
    """Shuffle multiple choice options while preserving the correct answer"""
    # strip off "A. ", "B. ", etc.
    texts = [c.split(". ", 1)[1] for c in doc["multiple_choices"]]
    # remember which was correct
    old_idx     = ord(doc["correct_answer"]) - ord("A")
    correct_txt = texts[old_idx]

    # Fisher–Yates shuffle
    for i in range(len(texts)-1, 0, -1):
        j = random.randint(0, i)
        texts[i], texts[j] = texts[j], texts[i]

    # re-label and recompute correct_answer
    new_choices = [f"{chr(65+i)}. {texts[i]}" for i in range(len(texts))]
    new_idx     = texts.index(correct_txt)
    new_correct = chr(65 + new_idx)

    return new_choices, new_correct

def shuffle_questions(collection, subject_filter):
    """Shuffle questions for a given subject"""
    query = {"subject": subject_filter}
    count = 0
    shuffled_count = 0

    for doc in collection.find(query):
        new_choices, new_correct = shuffle_doc(doc)

        # only write back if it changed
        if new_correct != doc["correct_answer"]:
            # Get the old and new correct text
            old_correct_text = doc["multiple_choices"][ord(doc["correct_answer"]) - ord("A")].split(". ", 1)[1]
            new_correct_text = new_choices[ord(new_correct) - ord("A")].split(". ", 1)[1]
            
            print(f"Old correct: {doc['correct_answer']} - {old_correct_text}")
            print(f"New correct: {new_correct} - {new_correct_text}")
            print("---")
            
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "multiple_choices": new_choices,
                    "correct_answer":   new_correct
                }}
            )
            shuffled_count += 1
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} documents...")

    print(f"Processed {count} questions total.")
    print(f"Shuffled {shuffled_count} questions.")
    return count, shuffled_count

def main():
    """Main method with hardcoded configuration"""
    # Hardcoded configuration
    prod_connection_string = "mongodb://admin:NewSecurePassword123!@3.128.97.182:27017/?authSource=admin"
    local_connection_string = "mongodb://localhost:27017"
    
    connection_string = prod_connection_string
    database_name = "adaptive_learning_docs"
    collection_name = "dryrun_questions"
    subject_filter = "AP Calculus AB"
    
    print(f"Connecting to MongoDB...")
    print(f"Connection: {connection_string}")
    print(f"Database: {database_name}")
    print(f"Collection: {collection_name}")
    print(f"Subject Filter: {subject_filter}")
    print("-" * 50)
    
    # Establish connection
    client = get_mongodb_connection(connection_string)
    if not client:
        print("Failed to establish MongoDB connection. Exiting.")
        return
    
    try:
        db = client[database_name]
        collection = db[collection_name]
        
        # Run the shuffle operation
        total_processed, total_shuffled = shuffle_questions(collection, subject_filter)
        
        print(f"\nOperation completed successfully!")
        print(f"Total questions processed: {total_processed}")
        print(f"Total questions shuffled: {total_shuffled}")
        
    except Exception as e:
        print(f"Error during shuffle operation: {e}")
    finally:
        client.close()
        print("MongoDB connection closed.")

if __name__ == "__main__":
    main()