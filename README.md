# 🧠 RAG Chatbot (Retrieval-Augmented Generation)

A simple **Persian-language chatbot** based on the **Retrieval-Augmented Generation (RAG)** architecture. It combines:
- A model to **retrieve relevant information** from a small knowledge base
- A language model to **generate natural language responses**

## 🚀 Features
- Supports Persian language queries and responses
- Uses Hugging Face Transformers for text generation
- Employs Sentence Transformers for semantic similarity
- Simple and educational structure for learning RAG concepts

## 🛠 Technologies Used
- `transformers` (for GPT-2 or compatible models)
- `sentence-transformers` (for sentence embeddings)
- `scikit-learn` (for cosine similarity)
- `PyTorch` (for model execution)

## 🧩 How It Works
1. **Retrieve relevant knowledge** from a simple knowledge base using semantic similarity.
2. **Build a context prompt** using the retrieved information and the user query.
3. **Generate a response** using a causal language model (e.g., GPT-2).

## 📦 Installation

```bash
pip install torch transformers sentence-transformers scikit-learn numpy







