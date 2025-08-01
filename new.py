from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')  # or whichever you're using
embedding = model.encode("test")
print(len(embedding))  # should print 1024
