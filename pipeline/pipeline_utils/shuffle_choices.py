import random
from pymongo import MongoClient

# 1) Configure connection
client     = MongoClient("mongodb://localhost:27017")
db         = client["adaptive_learning_docs"]
#collection = db["sat_math_collection"]
collection = db["dryrun_questions"]
# 2) Define your shuffle logic
def shuffle_doc(doc):
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

# 3) Run through only Algebra questions
query = {"subject":"AP Chemistry"}
count = 0

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
    count += 1
    print(f"Processed {count} documents...")

print(f"Processed and reshuffled {count} Algebra questions.")