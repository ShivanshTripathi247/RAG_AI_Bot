# AI-Powered RAG Chatbot for E-commerce

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.4+-green.svg)](https://www.mongodb.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple.svg)](https://www.pinecone.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains the source code for an advanced AI chatbot built for an e-commerce platform.  
It uses a **Retrieval-Augmented Generation (RAG)** architecture with the **Google Gemini API** and a **Pinecone vector store** to provide accurate, context-aware responses based on the business's own data.

---

## 🛠️ Core Technologies
- **Backend**: FastAPI (Python), LangChain  
- **Frontend**: ReactJS (Vite)  
- **AI & Data**: Google Gemini, Pinecone, Hugging Face (for embeddings), MongoDB  

---

## 🚀 Getting Started
Follow these steps to set up and run the project locally.

### Prerequisites
- Python 3.10+  
- Node.js 18+  
- A MongoDB database  
- API keys for **Google Gemini**, **Hugging Face**, and **Pinecone**  

---

### 1. Installation

First, clone the repository and install the dependencies for both the frontend and backend.

```bash
# 1. Clone the repository
git clone https://github.com/ShivanshTripathi247/RAG_AI_Bot.git
cd RAG_AI_Bot

# 2. Set up the backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt


```

---

### 2. Environment Setup

You will need to create `.env` files to store your API keys and configuration variables.

#### Backend (`backend/.env`)

Create a `.env` file in the `/backend` directory and add the following keys
(**do not commit this file to version control**):

```env
GOOGLE_API_KEY=your_google_api_key
HF_API_TOKEN=your_huggingface_api_token
EMBEDDING_ENDPOINT_URL=your_hf_embedding_endpoint
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
MONGO_URI=your_mongo_uri
DB_NAME=test
PRODUCTS_COLLECTION_NAME=products
ORDERS_COLLECTION_NAME=orders
FRONTEND_API_URL=http://localhost:5173
```

#### Frontend (`frontend/.env.local`)

Create a `.env.local` file in the `/frontend` directory and add the following key:

```env
VITE_RAGBOT_API_URL=http://127.0.0.1:8000
```

---

### 3. Running the Application

1. **Populate the Vector Database**
   Before the first run, you need to load your data into Pinecone. From the `/backend` directory (with your virtual environment active), run:

   ```bash
   python build_knowledge_base.py
   ```

2. **Start the Backend Server**
   From the `/backend` directory:

   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```



Your application is now running 🎉

---

## 📄 Note on Integration

This repository contains the source code for the RAG chatbot backend and a sample React frontend component only.
It is designed to be a standalone service that can be integrated into any existing e-commerce website.

The **full frontend and backend** of your primary e-commerce application will be unique to your project.
This chatbot is intended to plug into that existing infrastructure.

---

## 📬 Contact

For any inquiries, collaborations, or questions about this project, feel free to reach out:

* **LinkedIn**: [Connect with me](https://www.linkedin.com/in/shivansh-tripathi-4ab52a246/)
* **Email**: [shivansht06@gmail.com](mailto:shivansht06@gmail.com)