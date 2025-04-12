import openai
import chromadb

openai.api_key = "YOUR_API_KEY"
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="questions_collection")

def embed_question(question_text, question_id, metadata):
    response = openai.Embedding.create(
        input=question_text,
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    collection.add(
        documents=[question_text],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[question_id]
    )
