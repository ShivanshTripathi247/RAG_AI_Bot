AI-Powered RAG Chatbot for E-commerce
This repository contains the complete source code for an advanced AI chatbot designed for an e-commerce platform. The chatbot leverages a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware responses to user queries about products, policies, and personal orders.

<!-- Replace with a URL to a screenshot of your chatbot -->

🚀 Features
Context-Aware Responses: Uses a RAG pipeline to ground answers in factual data, preventing LLM hallucinations.

Structured Product Display: Displays product search results in clean, interactive cards instead of a wall of text.

Personalized Experience: Fetches real-time order history for logged-in users to provide personalized support.

Real-Time Knowledge Updates: Utilizes MongoDB Change Streams to automatically update the vector database whenever products are added or modified.

Dual User Modes: Seamlessly handles sessions for both guest users (general queries) and authenticated users (personalized queries).

Scalable & Lightweight Backend: The core backend is designed to be a lightweight orchestrator, offloading heavy ML tasks to dedicated services for high performance and low cost.

🛠️ Architecture & Tech Stack
This project uses a modern, decoupled architecture to ensure scalability and maintainability.

Backend: FastAPI (Python)

Frontend: ReactJS (with Vite)

Primary LLM: Google Gemini API (gemini-1.5-flash) for response generation.

Embedding Model: Hugging Face Inference Endpoint (sentence-transformers/all-MiniLM-L6-v2).

Vector Database: Pinecone (cloud-hosted) for efficient similarity search.

Primary Database: MongoDB for storing product, user, and order data.

Deployment: The backend is configured for deployment on cloud platforms like GCP, Render, or Hugging Face Spaces.

⚙️ Setup and Installation
Follow these steps to set up and run the project locally.

Prerequisites
Python 3.10+

Node.js 18+

A MongoDB database and its connection URI.

API keys for:

Google Gemini

Hugging Face (with write permissions)

Pinecone

1. Backend Setup
First, set up the Python backend server.

# Clone the repository
git clone <your-repository-url>
cd <repository-name>/backend  # Navigate to your backend folder

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt

Next, create the environment variables file.

Create a file named .env in the backend's root directory.

Add the following keys, replacing the placeholder values with your actual credentials:

# Gemini API Key
GOOGLE_API_KEY="AIza..."

# Hugging Face API Key
HF_API_TOKEN="hf_..."

# Hugging Face Endpoint URL for the EMBEDDING MODEL
EMBEDDING_ENDPOINT_URL="[https://your-embedding-endpoint.aws.endpoints.huggingface.cloud](https://your-embedding-endpoint.aws.endpoints.huggingface.cloud)"

# Pinecone Credentials
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="rag-bot" # Or your chosen index name

# MongoDB Credentials
MONGO_URI="mongodb+srv://..."
DB_NAME="test"
PRODUCTS_COLLECTION_NAME="products"
ORDERS_COLLECTION_NAME="orders"

# Frontend URL (for CORS)
FRONTEND_API_URL="http://localhost:5173"

2. Frontend Setup
Now, set up the React frontend.

# Navigate to your frontend folder from the root directory
cd ../frontend

# Install the required dependencies
npm install

Next, create the environment variables file for the frontend.

Create a file named .env.local in the frontend's root directory.

Add the following key, pointing to your local backend server:

VITE_RAGBOT_API_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)

▶️ Running the Application
Step 1: Populate the Vector Database
Before you can run the application, you must populate your Pinecone index with your product and website data.

Ensure you have your static content (e.g., about_us.txt) in the backend/data directory.

From the backend directory, run the build script:

# Make sure your venv is active
python build_knowledge_base.py

This script will connect to your MongoDB, read your products, load your text files, and upload everything to Pinecone.

Step 2: Start the Backend Server
With the vector database populated, you can start the API server.

# From the backend directory with venv active
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

Your backend is now running at http://127.0.0.1:8000.

Step 3: Start the Frontend Application
Finally, start the React development server.

# From the frontend directory
npm run dev

Your application should now be accessible in your browser, typically at http://localhost:5173. The chatbot will be fully functional and connected to your local backend.

☁️ Deployment
Backend: The backend is configured for deployment on cloud services like Render or GCP Compute Engine. The main.py file includes a health check endpoint at /health for platform compatibility.

Frontend: The React application can be deployed on static hosting platforms like Vercel or Netlify. Remember to update the VITE_RAGBOT_API_URL environment variable to point to your live backend's URL.