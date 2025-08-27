import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_knowledge_base():
    """
    Connects to MongoDB, fetches product data, loads static text files,
    and builds a combined FAISS vector store.
    """
    # 1. Load Environment Variables
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")

    if not mongo_uri:
        logging.error("MONGO_URI not found in .env file. Please add it.")
        return

    # --- IMPORTANT: UPDATE THESE VALUES ---
    db_name = "test"
    collection_name = "products"
    # ------------------------------------

    # 2. Connect to MongoDB and Fetch Products
    try:
        logging.info("Connecting to MongoDB...")
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        products = list(collection.find({}))
        logging.info(f"Found {len(products)} products in MongoDB.")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB or fetch products: {e}")
        return

    # 3. Format MongoDB Documents for LangChain
    mongo_documents = []
    for product in products:
        page_content = (
            f"Product Name: {product.get('title', 'N/A')}. "
            f"Description: {product.get('description', 'N/A')}. "
            f"Price: ₹{product.get('price', 'N/A')}. "
            f"Sale Price: ₹{product.get('salePrice', 'N/A')}. "
            f"Category: {product.get('category', 'N/A')}. "
            f"Stock: {product.get('totalStock', 'N/A')} units available."
        )
        metadata = {
            "source": "mongodb_products",
            "product_id": str(product.get('_id')),
            "image_url": product.get('image', ''),
            "price": product.get('price', 0),
            "salePrice": product.get('salePrice', 0)
        }
        doc = Document(page_content=page_content, metadata=metadata)
        mongo_documents.append(doc)
    logging.info(f"Formatted {len(mongo_documents)} documents from MongoDB.")

    # --- NEW SECTION: LOAD STATIC FILES ---
    try:
        logging.info("Loading documents from the 'data' directory...")
        # This looks for any .txt file in the ./data/ folder
        loader = DirectoryLoader('./data/', glob="**/*.txt")
        file_documents = loader.load()
        for doc in file_documents:
            doc.metadata['source'] = 'static_file' # Add source metadata for context
        logging.info(f"Loaded {len(file_documents)} documents from files.")
    except Exception as e:
        logging.error(f"Failed to load static text files: {e}")
        file_documents = [] # Ensure it's a list even if it fails
    # ------------------------------------

    # 4. Combine all documents into a single list
    all_documents = mongo_documents + file_documents
    if not all_documents:
        logging.error("No documents to process. Exiting.")
        return
    logging.info(f"Total documents to be indexed: {len(all_documents)}")

    # 5. Embed and Store in FAISS
    try:
        logging.info("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        logging.info("Creating FAISS vector store from all documents. This may take a while...")
        # Use the combined list of all documents
        vector_store = FAISS.from_documents(all_documents, embedding_model)
        
        # 6. Save to Disk
        vector_store.save_local("faiss_index")
        logging.info("✅ Knowledge base updated successfully and saved to the 'faiss_index' folder.")
    except Exception as e:
        logging.error(f"An error occurred during embedding or saving the vector store: {e}")

if __name__ == "__main__":
    create_knowledge_base()