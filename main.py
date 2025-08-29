import os
import logging
import re
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from operator import itemgetter

from pinecone import Pinecone as PineconeClient
from langchain_pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings # New embedder
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter # Added for change stream
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings

# --- NEW: Import ChatPromptTemplate for system instructions ---
from langchain.prompts import ChatPromptTemplate, PromptTemplate
# -----------------------------------------------------------

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Imports for Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
# -------------------------

# --- 1. INITIAL SETUP ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_ENDPOINT_URL = os.getenv("EMBEDDING_ENDPOINT_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize global variables as None
embedding_model = None
vector_store = None
retriever = None
llm = None
rag_chain = None
mongo_client = None
db = None
orders_collection = None
products_collection = None # Added for change stream
pinecone_index = None # Added for direct access to the Pinecone index

# --- NEW: Renamed THREAD LOCK FOR CLARITY ---
vector_store_lock = threading.Lock()
# ---------------------------------------------

# --- 2. FASTAPI APPLICATION DEFINITION ---
app = FastAPI(title="E-commerce AI Chatbot", description="An API for the e-commerce RAG chatbot.")

origins = [
    os.getenv("FRONTEND_API_URL"),
    os.getenv("BACKEND_API_URL"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in origins if origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. STARTUP EVENT TO LOAD MODELS ---
@app.on_event("startup")
async def startup_event():
    global embedding_model, vector_store, retriever, llm, rag_chain, mongo_client, db, orders_collection, products_collection, pinecone_index

    logging.info("Application startup: Initializing services...")
    
    try:
        # Embedding model and Pinecone connection
        embedding_model = HuggingFaceEndpointEmbeddings(model=EMBEDDING_ENDPOINT_URL, huggingfacehub_api_token=HF_API_TOKEN)
        pc = PineconeClient(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME) # Get a direct handle to the index
        vector_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embedding_model)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Initialize Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=GOOGLE_API_KEY,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
            convert_system_message_to_human=True
        )
    
        # MongoDB connection
        DB_NAME = "test"
        PRODUCTS_COLLECTION_NAME = "products"
        ORDERS_COLLECTION_NAME = "orders"
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client[DB_NAME]
        products_collection = db[PRODUCTS_COLLECTION_NAME]
        orders_collection = db[ORDERS_COLLECTION_NAME]
        
        # --- PROMPT & RAG CHAIN DEFINITION ---
        system_instruction = """Your name is DaVinci. You are a helpful and friendly e-commerce assistant for a platform named Idezign Studio.
        Your goal is to help users with their questions about products and their orders in details, with enthusiasm.
        Use only the context provided to answer the user's question. If you are providing product image link in the response please provide it with a text "**click here to view**" with markdown and highlighting. 
        Always answer the question with proper markdown formatting, headings, text highlighting, specially the title of the product highlighted like **Product Name**.
        If the questions are not related to e-commerce or Interior Decor, politely say that you are not capable of answering that.
        If you don't know the answer from the context, say that you don't have enough information to answer."""

        human_prompt_template = """
        **General Knowledge Context:**
        {general_context}

        **User's Recent Order History (if available):**
        {order_context}

        **User's Question:**
        {question}
        """

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", human_prompt_template),
        ])

        retrieval_chain = {"general_context": itemgetter("question") | retriever, "question": itemgetter("question"), "order_context": itemgetter("order_context")}
        rag_chain = ( retrieval_chain | chat_prompt | llm | StrOutputParser() )
        
        logging.info("Initialization complete. Services are ready.")

        # --- START THE BACKGROUND TASK ---
        listener_thread = threading.Thread(target=listen_for_product_changes, daemon=True)
        listener_thread.start()
        # -------------------------------------------
        
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize services during startup: {e}", exc_info=True)
        rag_chain = None

# --- HELPER FUNCTION TO FORMAT PRODUCTS ---
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
    }
    return Document(page_content=page_content, metadata=metadata)
# -----------------------------------------------

# --- CORRECTED MONGODB CHANGE STREAM LISTENER ---

# # --- OLD, INCORRECT LISTENER (Designed for FAISS) ---
# def listen_for_product_changes():
#     logging.info("Starting MongoDB change stream listener...")
#     try:
#         with products_collection.watch(full_document='updateLookup') as stream:
#             for change in stream:
#                 operation_type = change['operationType']
#                 logging.info(f"Change detected in products collection: {operation_type}")
#
#                 with faiss_lock: # Using a generic lock name, as it's for the vector store
#                     if operation_type in ['insert', 'update']:
#                         doc_id = str(change['documentKey']['_id'])
#                         
#                         # For Pinecone, we just upsert with the same ID to update
#                         new_doc = format_product_for_langchain(change['fullDocument'])
#                         vector_store.add_documents([new_doc], ids=[doc_id])
#                         logging.info(f"Upserted product in index: {change['fullDocument'].get('title', 'N/A')}")
#
#                     elif operation_type == 'delete':
#                         doc_id = str(change['documentKey']['_id'])
#                         vector_store.delete([doc_id])
#                         logging.info(f"Deleted product from index with ID: {doc_id}")
#     except Exception as e:
#         logging.error(f"Error in MongoDB change stream: {e}", exc_info=True)
# # ----------------------------------------------------


# --- NEW, CORRECTED LISTENER (For Pinecone) ---
def listen_for_product_changes():
    logging.info("Starting MongoDB change stream listener for Pinecone...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
    
    try:
        with products_collection.watch(full_document='updateLookup') as stream:
            for change in stream:
                operation_type = change['operationType']
                doc_id = str(change['documentKey']['_id'])
                logging.info(f"Change detected for product {doc_id}: {operation_type}")

                with vector_store_lock:
                    # For both insert and update, we delete the old chunks and upsert the new ones
                    if operation_type in ['insert', 'update']:
                        # Delete existing chunks for this product to handle updates correctly
                        pinecone_index.delete(filter={"product_id": doc_id})

                        # Format, chunk, embed, and upsert the new document
                        new_doc = format_product_for_langchain(change['fullDocument'])
                        doc_chunks = text_splitter.split_documents([new_doc])
                        
                        texts_to_embed = [chunk.page_content for chunk in doc_chunks]
                        embeddings = embedding_model.embed_documents(texts_to_embed)

                        vectors_to_upsert = []
                        for i, chunk in enumerate(doc_chunks):
                            chunk_id = f"{doc_id}_chunk_{i}"
                            vectors_to_upsert.append({
                                "id": chunk_id,
                                "values": embeddings[i],
                                "metadata": {"text": chunk.page_content, **chunk.metadata}
                            })
                        
                        if vectors_to_upsert:
                            pinecone_index.upsert(vectors=vectors_to_upsert)
                            logging.info(f"Upserted {len(vectors_to_upsert)} chunks for product: {doc_id}")

                    elif operation_type == 'delete':
                        # Delete all chunks associated with this product_id
                        pinecone_index.delete(filter={"product_id": doc_id})
                        logging.info(f"Deleted all chunks for product from index with ID: {doc_id}")

    except Exception as e:
        logging.error(f"Error in MongoDB change stream: {e}", exc_info=True)
# ------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok"}

class Query(BaseModel):
    question: str
    user_id: Optional[str] = Field(None, description="Optional user ID for personalized queries")

def fetch_user_orders(user_id: str) -> str:
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

@app.post("/ask", summary="Ask the chatbot a question")
def ask_question(query: Query):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Service is warming up or has failed to initialize. Please try again in a moment.")
    
    question = query.question
    user_id = query.user_id
    image_url = ""

    # Using a lock to ensure thread-safe reads from the vector store
    with vector_store_lock: 
        if user_id:
            order_context = fetch_user_orders(user_id)
            response = rag_chain.invoke({"order_context": order_context, "question": question})
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
                
                response = rag_chain.invoke({
                    "general_context": general_context,
                    "order_context": "User is not logged in.",
                    "question": question
                })
                response_type = "answer"

    return {"response_type": response_type, "answer": response, "image_url": image_url}

