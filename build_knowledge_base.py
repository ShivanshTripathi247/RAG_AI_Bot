import os
import logging
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.schema.document import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter # <-- NEW IMPORT
from pinecone import Pinecone as PineconeClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_knowledge_base():
    """
    Connects to data sources, chunks documents, generates embeddings,
    and upserts them to a Pinecone index.
    """
    # --- 1. Load Environment Variables and Data ---
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    hf_api_token = os.getenv("HF_API_TOKEN")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    embedding_endpoint_url = os.getenv("EMBEDDING_ENDPOINT_URL")

    # Fetch and format all documents from MongoDB and local files
    client = MongoClient(mongo_uri)
    db = client[os.getenv("DB_NAME", "test")]
    collection = db[os.getenv("PRODUCTS_COLLECTION_NAME", "products")]
    products = list(collection.find({}))
    
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
    
    file_documents = DirectoryLoader('./data/', glob="**/*.txt").load()
    for doc in file_documents:
        doc.metadata['source'] = 'static_file'

    all_documents = mongo_documents + file_documents
    logging.info(f"Found {len(all_documents)} total documents to process.")

    # --- NEW: SPLIT DOCUMENTS INTO SMALLER CHUNKS ---
    logging.info("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
    doc_chunks = text_splitter.split_documents(all_documents)
    logging.info(f"Split {len(all_documents)} documents into {len(doc_chunks)} chunks.")
    # ------------------------------------------------

    try:
        # --- 2. Initialize Clients ---
        logging.info("Initializing embedding model and Pinecone client...")
        embedding_model = HuggingFaceEndpointEmbeddings(
            model=embedding_endpoint_url,
            huggingfacehub_api_token=hf_api_token
        )
        pc = PineconeClient(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)

        # --- 3. Generate Embeddings for the Chunks ---
        logging.info("Generating embeddings for all document chunks...")
        texts_to_embed = [chunk.page_content for chunk in doc_chunks]
        embeddings = embedding_model.embed_documents(texts_to_embed)
        logging.info(f"Successfully generated {len(embeddings)} embeddings.")

        # --- 4. Manually Upsert Chunks to Pinecone ---
        logging.info("Preparing data and upserting to Pinecone...")
        vectors_to_upsert = []
        for i, chunk in enumerate(doc_chunks):
            # Create a unique ID for each chunk
            vector_id = f"chunk_{i}_{os.urandom(4).hex()}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embeddings[i],
                "metadata": { "text": chunk.page_content, **chunk.metadata }
            })

        # Upsert data in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
            logging.info(f"Upserted batch {i//batch_size + 1}...")
        
        logging.info("Upsert process completed.")
        
        # --- 5. Wait and Verify ---
        logging.info("Waiting 10 seconds for the Pinecone index to update...")
        time.sleep(10)

        stats = index.describe_index_stats()
        final_count = stats.get('total_vector_count', 0)
        logging.info(f"Final verification - Vector count in Pinecone: {final_count}")

        if final_count == len(doc_chunks):
            logging.info("✅ Success! Knowledge base has been populated correctly.")
        else:
            logging.error(f"❌ Mismatch after upsert. Expected {len(doc_chunks)} vectors, but found {final_count}.")

    except Exception as e:
        logging.error(f"An error occurred during the build process: {e}", exc_info=True)

if __name__ == "__main__":
    create_knowledge_base()