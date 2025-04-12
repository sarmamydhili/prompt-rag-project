import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.get_collection(name="questions_collection")

def find_similar_questions(query_text, n_results=3):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results
