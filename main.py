import os
import logging
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from operator import itemgetter

from fastapi.middleware.cors import CORSMiddleware
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. INITIAL SETUP ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")
MONGO_URI = os.getenv("MONGO_URI")

# --- IMPORTANT: UPDATE THESE VALUES ---
DB_NAME = "test"
PRODUCTS_COLLECTION_NAME = "products" # <-- NEW: Added collection name for products
ORDERS_COLLECTION_NAME = "orders"
# ------------------------------------

# Initialize models and retriever
try:
    logging.info("Initializing models and loading knowledge base...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = HuggingFaceEndpoint(
        endpoint_url=HF_ENDPOINT_URL,
        huggingfacehub_api_token=HF_API_TOKEN,
        task="text-generation",
        max_new_tokens=512,
        return_full_text=False
    )
    logging.info("Initialization complete.")
except Exception as e:
    logging.error(f"Failed to initialize models or load vector store: {e}")
    raise

# Connect to MongoDB
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DB_NAME]
    products_collection = db[PRODUCTS_COLLECTION_NAME] # <-- NEW: Connect to products collection
    orders_collection = db[ORDERS_COLLECTION_NAME]
    logging.info("Successfully connected to MongoDB.")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    raise

# --- NEW: THREAD LOCK FOR SAFE INDEX UPDATES ---
faiss_lock = threading.Lock()
# ---------------------------------------------

# --- 2. PROMPT & RAG CHAIN DEFINITION (No changes here) ---
prompt_template = """
Your name is DaVinci. You are a helpful and friendly e-commerce/interior-decor assistant. Name of the e-commerce/interior-decor platform is Idezign Studio. Your goal is to help users with their questions about products and their orders.
Use the following context to answer the user's question. Always answer the question with proper markdown formatting. If you don't know the answer from the context, say you don't have that information.

**General Knowledge Context:**
{general_context}

**User's Recent Order History (if available):**
{order_context}

**User's Question:**
{question}

Helpful Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)
retrieval_chain = {
    "general_context": itemgetter("question") | retriever,
    "question": itemgetter("question"),
    "order_context": itemgetter("order_context")
}
rag_chain = ( retrieval_chain | prompt | llm | StrOutputParser() )

# --- 3. HELPER FUNCTIONS ---

def fetch_user_orders(user_id: str) -> str:
    # (No changes to this function)
    try:
        recent_orders = list(orders_collection.find({"userId": user_id}).sort("orderDate", -1).limit(3))
        if not recent_orders:
            return "No recent orders found for this user."
        order_context_str = "Here are your most recent orders:\n"
        for order in recent_orders:
            items_str = ", ".join([item.get('title', 'N/A') for item in order.get('cartItems', [])])
            order_context_str += (
                f"- Order ID: {order.get('_id')}, "
                f"Status: {order.get('orderStatus', 'N/A')}, "
                f"Items: {items_str}, "
                f"Total: ${order.get('totalAmount', 'N/A')}\n"
            )
        return order_context_str
    except Exception as e:
        logging.error(f"Error fetching orders for user {user_id}: {e}")
        return "There was an error retrieving your order history."

# --- NEW: HELPER FUNCTION TO FORMAT PRODUCTS ---
def format_product_for_langchain(product: dict) -> Document:
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
    return Document(page_content=page_content, metadata=metadata)
# -----------------------------------------------

# --- NEW: MONGODB CHANGE STREAM LISTENER ---
def listen_for_product_changes():
    logging.info("Starting MongoDB change stream listener...")
    try:
        with products_collection.watch(full_document='updateLookup') as stream:
            for change in stream:
                operation_type = change['operationType']
                logging.info(f"Change detected in products collection: {operation_type}")

                with faiss_lock:
                    if operation_type in ['insert', 'update']:
                        doc_id = str(change['documentKey']['_id'])
                        # For updates, we delete the old entry first
                        if operation_type == 'update':
                            # This is a simplified delete; requires iterating to find the doc to delete
                            ids_to_delete = [
                                i for i, doc in vector_store.docstore._dict.items() 
                                if doc.metadata.get("product_id") == doc_id
                            ]
                            if ids_to_delete:
                                vector_store.delete(ids_to_delete)
                                logging.info(f"Deleted old version of product ID {doc_id} for update.")
                        
                        new_doc = format_product_for_langchain(change['fullDocument'])
                        vector_store.add_documents([new_doc])
                        logging.info(f"Added/Updated product in index: {change['fullDocument'].get('title', 'N/A')}")

                    elif operation_type == 'delete':
                        doc_id = str(change['documentKey']['_id'])
                        ids_to_delete = [
                            i for i, doc in vector_store.docstore._dict.items() 
                            if doc.metadata.get("product_id") == doc_id
                        ]
                        if ids_to_delete:
                            vector_store.delete(ids_to_delete)
                            logging.info(f"Deleted product from index with ID: {doc_id}")
                    
                    vector_store.save_local("faiss_index")
                    logging.info("FAISS index updated and saved to disk.")
    except Exception as e:
        logging.error(f"Error in MongoDB change stream: {e}", exc_info=True)
# ------------------------------------------

# --- 4. FASTAPI APPLICATION ---
app = FastAPI(title="E-commerce AI Chatbot", description="An API for the e-commerce RAG chatbot.")

origins = [
    "http://localhost:5173", # Your React frontend's origin
    "http://localhost:3000", # Another common React port
    # Add your production frontend URL here when you deploy
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- NEW: START THE BACKGROUND TASK ON APP STARTUP ---
@app.on_event("startup")
async def startup_event():
    listener_thread = threading.Thread(target=listen_for_product_changes, daemon=True)
    listener_thread.start()
# ----------------------------------------------------

class Query(BaseModel):
    question: str
    user_id: Optional[str] = Field(None, description="Optional user ID for personalized queries")

@app.post("/ask", summary="Ask the chatbot a question")
def ask_question(query: Query):
    question = query.question
    user_id = query.user_id
    image_url = ""

    with faiss_lock: # Use lock to ensure we read a stable index
        if user_id:
            order_context = fetch_user_orders(user_id)
            general_context = "\n".join([doc.page_content for doc in retriever.invoke(question)])
            response = rag_chain.invoke({"general_context": general_context, "order_context": order_context, "question": question})
            response_type = "answer"
        else:
            login_keywords = ["my order", "my account", "track package", "where is my stuff"]
            if any(keyword in question.lower() for keyword in login_keywords):
                response = "It looks like you're asking about a personal order. Please log in to your account, and I'll be able to help you with that!"
                response_type = "login_prompt"
            else:
                retrieved_docs = retriever.invoke(question)
                general_context = "\n".join([doc.page_content for doc in retrieved_docs])
                if retrieved_docs:
                    image_url = retrieved_docs[0].metadata.get("image_url", "")
                response = rag_chain.invoke({"general_context": general_context, "order_context": "User is not logged in.", "question": question})
                response_type = "answer"

    return {"response_type": response_type, "answer": response, "image_url": image_url}