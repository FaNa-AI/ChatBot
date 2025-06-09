import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. Model and Tokenizer Setup ---
# Text generation model - Flan-T5-Base
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model to CPU or GPU depending on availability
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float32
)

# Sentence embedding model for semantic similarity
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- 2. Knowledge Base ---
# Simple in-memory knowledge base (could be loaded from files/databases in real use cases)
knowledge_base = [
    "Python is a high-level interpreted programming language.",
    "Machine learning is a subset of AI that allows computers to learn from data without being explicitly programmed.",
    "Deep learning is a type of machine learning that uses deep neural networks.",
    "RAG (Retrieval-Augmented Generation) is an architecture that combines retrieval and text generation.",
    "RAG enables language models to access external information and provide more accurate responses.",
    "TensorFlow and PyTorch are two popular deep learning frameworks.",
    "Chatbots are computer programs that simulate human conversation.",
    "Large Language Models (LLMs) are trained on massive amounts of text and can generate text, translate, and answer questions.",
    "Artificial intelligence is a field of computer science focused on creating intelligent machines.",
    "Germany is located in central Europe and its capital is Berlin.",
    "Germany is known for its automotive industry and precision engineering.",
    "Frankfurt is Germany’s financial hub and home to the European Central Bank.",
    "Berlin, the capital of Germany, is a city rich in history and culture."
]

# Generate embeddings for knowledge base
knowledge_embeddings = sentence_model.encode(knowledge_base, convert_to_tensor=True)

# --- 3. Retriever Function ---
def retrieve_relevant_docs(query, top_k=2):
    """
    Retrieve the most semantically similar documents from the knowledge base.
    """
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), knowledge_embeddings.cpu().numpy())
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    relevant_docs = [knowledge_base[i] for i in top_indices]
    return relevant_docs

# --- 4. Generator Function ---
def generate_response(prompt, max_length=100):
    """
    Generate a response using the Flan-T5-Base language model.
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")
    else:
        model.to("cpu")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# --- 5. Main RAG Chatbot Function ---
def rag_chatbot(query):
    """
    Executes the full RAG pipeline to respond to a user query.
    """
    relevant_docs = retrieve_relevant_docs(query)
    print(f"Retrieved documents: {relevant_docs}")

    context = "\n".join(relevant_docs)
    prompt = f"Consider the following text:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = generate_response(prompt)
    return response

# --- 6. Example Usage ---
if __name__ == "__main__":
    print("Welcome to the RAG Chatbot using Flan-T5-Base! Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break

        response = rag_chatbot(user_query)
        print(f"Chatbot: {response}")

        response = rag_chatbot(user_query)
        print(f"چت‌بات: {response}")
