import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# basic setup
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")
proj_id = os.getenv("OPENAI_PROJECT_ID")

# create openai client 
client = OpenAI(
    api_key=api_key,
    organization=org_id,
    project=proj_id
)

# extract text from pdf
def extract_text_from_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

# chunk the text
def split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# get embeddings for each chunk
def get_embeddings(chunks):
    embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"[Chunk {i}] Failed to generate embedding: {e}")
            embeddings.append(np.zeros(1536))  # use zero vector for failure case
    return np.array(embeddings)

# Build FAISS index
def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ask question 
def ask_question(query, chunks, index, embeddings):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = np.array(response.data[0].embedding).reshape(1, -1)
    except Exception as e:
        return f"Error generating embedding for your question: {e}"

    top_k = 3
    distances, indices = index.search(query_embedding, top_k)

    context = " ".join([chunks[i] for i in indices[0]])

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for answering questions about a PDF document."},
                {"role": "user", "content": f"Here is some context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting response from GPT: {e}"

# MAIN EXECUTION 
if __name__ == "__main__":
    pdf_path = "your_file.pdf"

    print("Reading PDF...")
    full_text = extract_text_from_pdf(pdf_path)

    if not full_text.strip():
        print("No text found in the PDF. Exiting.")
        exit()

    print("Splitting into chunks...")
    chunks = split_text(full_text)

    print("Generating embeddings...")
    embeddings = get_embeddings(chunks)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("\nReady to chat! Type your question below:")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = ask_question(query, chunks, index, embeddings)
        print("Chatbot:", answer)
