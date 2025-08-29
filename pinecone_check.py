import os
from dotenv import load_dotenv
from pinecone import Pinecone

def check_index_status():
    """
    Connects to Pinecone and fetches the status of the specified index.
    """
    load_dotenv()

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME]):
        print("Error: Make sure PINECONE_API_KEY and PINECONE_INDEX_NAME are set in your .env file.")
        return

    try:
        print("Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)

        print(f"Checking status of index: '{PINECONE_INDEX_NAME}'...")

        # Check if the index exists
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print(f"Error: Index '{PINECONE_INDEX_NAME}' does not exist.")
            return

        # Get the index object
        index = pc.Index(PINECONE_INDEX_NAME)

        # Fetch and print the statistics
        stats = index.describe_index_stats()
        print("\n--- Index Statistics ---")
        print(f"Vector Count: {stats['total_vector_count']}")
        print(f"Dimension: {stats['dimension']}")
        print("------------------------\n")

        if stats['total_vector_count'] > 0:
            print("✅ Success! Your index is populated.")
        else:
            print("❌ Your index is empty. The build script may not have run correctly.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_index_status()