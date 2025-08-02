from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')  # Standardized to BGE-large
embedding = model.encode("test")
print(len(embedding))  # should print 1024 for BGE-large
