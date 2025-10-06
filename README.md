# 📚 Manvi.AI — Chat with Multiple PDFs using ChatGroq & LangChain

Manvi.AI is an intelligent **PDF chatbot** built with **Streamlit**, **LangChain**, and **ChatGroq**, allowing you to upload multiple PDF documents, process their content, and interact with them conversationally using natural language.  

It uses **FAISS** for efficient semantic search, **HuggingFace Embeddings** for text vectorization, and **ChatGroq** for powerful LLM-based reasoning.

---

## 🚀 Features

- 🧠 **Multi-PDF Support:** Upload and query multiple PDFs at once.
- ⚡ **Conversational Memory:** Keeps track of previous interactions for contextual responses.
- 🔍 **Semantic Search:** Retrieves the most relevant document chunks using FAISS.
- 💬 **Real-time QA:** Ask any question about your uploaded documents.
- 🧩 **Streamlit Interface:** Intuitive and interactive UI for document chat.
- 🔒 **Secure API Key Management:** Supports both `.env` (local) and Streamlit Secrets (deployment).

---

## 🏗️ Tech Stack

| Component | Library / Framework |
|------------|--------------------|
| Frontend | Streamlit |
| Backend | LangChain |
| LLM | ChatGroq (`openai/gpt-oss-120b`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Database | FAISS |
| Memory | ConversationBufferMemory |
| File Reader | PyPDF2 |

---

## 🧩 Project Structure

ManviAI/
│
├── app.py # Main Streamlit app
├── htmlTemplates.py # HTML/CSS templates for chat styling
├── requirements.txt # Dependencies list
├── .env # (Optional) Local environment variables
└── .streamlit/
└── secrets.toml # Streamlit Cloud secrets (for deployment)
