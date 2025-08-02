import chromadb
from chromadb.config import Settings
from config import Config


def print_collection_dimensions():
    # Try to connect to persistent client
    try:
        client = chromadb.PersistentClient(path=Config.VECTOR_STORE_PATH)
    except Exception as e:
        print(f"Warning: PersistentClient failed: {e}")
        print("Falling back to in-memory ChromaDB client (will not show persistent collections)")
        client = chromadb.Client()

    collections = client.list_collections()
    if not collections:
        print("No collections found.")
        return

    print("Found collections:")
    for coll in collections:
        try:
            collection = client.get_collection(name=coll.name)
            results = collection.get(limit=1)
            print(f"Sample get() result for {coll.name}: {results}")  # Add this line
            dim = None
            if results and 'embeddings' in results and results['embeddings']:
                dim = len(results['embeddings'][0])
            print(f"- {coll.name}: embedding dimension = {dim if dim else 'Unknown'}")
        except Exception as e:
            print(f"- {coll.name}: Error retrieving dimension ({e})")

if __name__ == "__main__":
    print_collection_dimensions()
