from langchain_ollama import OllamaEmbeddings
import numpy as np

print("1. Creating embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

text = "Java Streams are used to process collections declaratively."

print("2. Creating embedding...")

vector = embeddings.embed_query(text)

print("\n3. Vector type:")
print(type(vector))

print("\n4. Vector dimension:")
print(len(vector))

print("\n5. First 10 values:")
print(vector[:10])

print("\n6. Embediings for Java Streams:")
print(embeddings.embed_query("Java Streams")[:10])

print("\n7. Embediings for Java Collections:")
print(embeddings.embed_query("Java Collections")[:10])

print("\n8. Embediings for weather:")
print(embeddings.embed_query("weather")[:10])

"""---------------------------------------------------"""

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

stream_vector = embeddings.embed_query("Java Streams")
collections_vector = embeddings.embed_query("Java Collections")
weather_vector = embeddings.embed_query("weather")

print("\nSimilarity: Java Streams vs Java Collections")
print(cosine_similarity(stream_vector, collections_vector))

print("\nSimilarity: Java Streams vs Weather")
print(cosine_similarity(stream_vector, weather_vector))

print("\nSimilarity: Java Collections vs Weather")
print(cosine_similarity(collections_vector, weather_vector))