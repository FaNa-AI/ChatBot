import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. Model and Tokenizer Setup ---
# Text generation model (Generator)
# You can use Persian models like 'HooshvareLab/gpt2-fa' or general models like 'distilgpt2'
# For this example, we use a small and fast model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add pad token if not already present to avoid errors
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Sentence embedding model (Retriever) for semantic similarity
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- 2. Knowledge Base ---
# This is a simple knowledge base. In real applications, it can be loaded from files, databases, etc.
knowledge_base = [
    "Python is a high-level interpreted programming language.",
    "Machine learning is a subset of artificial intelligence that allows computers to learn from data without explicit programming.",
    "Deep learning is a type of machine learning that uses deep neural networks.",
    "RAG (Retrieval-Augmented Generation) is an architecture that combines information retrieval with text generation.",
    "RAG enables language models to access external information and provide more accurate answers.",
    "TensorFlow and PyTorch are two popular frameworks for deep learning.",
    "Chatbots are computer programs that simulate human conversation.",
    "Large Language Models (LLMs) are trained on vast amounts of text data and can generate text, translate, and answer questions.",
    "Artificial Intelligence is a field of computer science that deals with building intelligent machines."
]

# Generate embeddings for the knowledge base
knowledge_embeddings = sentence_model.encode(knowledge_base, convert_to_tensor=True)

# --- 3. Retriever Function ---
def retrieve_relevant_docs(query, top_k=2):
    """
    Retrieve the most semantically relevant documents from the knowledge base.
    """
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), knowledge_embeddings.cpu().numpy())
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    relevant_docs = [knowledge_base[i] for i in top_indices]
    return relevant_docs

# --- 4. Generator Function ---
def generate_response(prompt, max_length=150):
    """
    Generate a response using the language model.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# --- 5. RAG Chatbot Function ---
def rag_chatbot(query):
    """
    Executes the full RAG process to respond to a query.
    """
    relevant_docs = retrieve_relevant_docs(query)
    print(f"Retrieved documents: {relevant_docs}")

    # Prepare the prompt with the retrieved documents
    context = "\n".join(relevant_docs)
    prompt = f"Consider the following text:\n{context}\n\nBased on this information and your own knowledge, answer the following question: {query}\nAnswer:"

    # Generate response
    response = generate_response(prompt)
    return response

# --- 6. Example Usage ---
if __name__ == "__main__":
    print("Welcome to the RAG Chatbot! Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = rag_chatbot(user_query)
        print(f"Chatbot: {response}")
