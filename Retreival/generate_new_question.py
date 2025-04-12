import openai
import chromadb
import config

# 1. Setup OpenAI and ChromaDB
openai.api_key = config.OPENAI_API_KEY
chroma_client = chromadb.Client()
collection = chroma_client.get_collection(name=config.CHROMA_COLLECTION_NAME)

# 2. Retrieve 3 Similar Questions
def retrieve_similar_questions(query_text, n_results=3):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results['documents'][0]  # Returns list of similar question texts

# 3. Generate New Question Using Retrieved Examples
def generate_new_question(similar_questions):
    examples_text = "\n\n".join([f"Example {i+1}: {q}" for i, q in enumerate(similar_questions)])

    prompt = f"""
You are an AI assistant trained to generate educational questions.

Here are some examples of existing questions:

{examples_text}

Based on the style and difficulty of the above examples, generate a **new** multiple-choice question on a related topic.

The question should be medium difficulty and suitable for a high school student.
Only output the question text without any explanation.
"""

    response = openai.ChatCompletion.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": "You generate high-quality educational questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=config.TEMPERATURE
    )

    new_question = response['choices'][0]['message']['content']
    return new_question.strip()

# 4. Main Flow
if __name__ == "__main__":
    # Example query: user types a topic
    query = "forces on inclined plane"  # You can modify this or take input from user

    print("\n🔎 Retrieving similar questions...")
    similar = retrieve_similar_questions(query)

    print("\n🎯 Similar Examples Retrieved:")
    for i, q in enumerate(similar):
        print(f"{i+1}. {q}\n")

    print("\n🧠 Generating new question based on these...")
    new_question = generate_new_question(similar)

    print("\n📝 New Generated Question:")
    print(new_question)
