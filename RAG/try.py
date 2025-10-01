import fitz  # from PyMuPDF
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tkinter as tk
from tkinter import filedialog

# ✅ Configure Gemini API
GEMINI_API_KEY = "AIzaSyAUJrMKFNR2YgkE22Orzufwo-tD2xggDVk"  # Replace with your actual Gemini API Key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# ✅ Extract text from uploaded PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# ✅ Split text into chunks
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ✅ Retrieve top-k relevant chunks using TF-IDF
def retrieve_context(query, chunks, top_k=3):
    vectorizer = TfidfVectorizer().fit(chunks + [query])
    vectors = vectorizer.transform(chunks + [query])
    query_vector = vectors[-1]
    chunk_vectors = vectors[:-1]
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return "\n".join([chunks[i] for i in top_indices])

# ✅ Ask Gemini with context
def ask_gemini(query, context):
    prompt = f"""
You are a helpful assistant.

Given the context from a document:
---------------------
{context}
---------------------

Answer this query:
"{query}"

Be concise and answer using only the given context.
"""
    response = model.generate_content(prompt)
    return response.text

# ✅ File picker GUI
def pick_pdf_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select PDF File",
        filetypes=[("PDF files", "*.pdf")]
    )
    return file_path

# ✅ Main logic
def main():
    print("📄 RAG using Gemini & PyMuPDF\n")
    file_path = pick_pdf_file()

    if not file_path:
        print("❌ No file selected.")
        return

    print(f"\n📂 Selected: {file_path}")
    print("🔄 Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(file_path)
    print(f"✅ Extracted {len(pdf_text)} characters.")

    chunks = chunk_text(pdf_text)

    print("\nYou can now ask questions about the PDF.")
    print("💬 Type 'exit' to quit.\n")

    while True:
        user_query = input("🔎 Your question: ").strip()
        if user_query.lower() == "exit":
            print("👋 Exiting... Have a great day!")
            break

        context = retrieve_context(user_query, chunks)

        print("\n📚 Relevant Context:\n")
        print(context)

        print("\n🤖 Gemini's Answer:\n")
        answer = ask_gemini(user_query, context)
        print(answer)
        print("-" * 80)

# ✅ Entry point
if __name__ == "__main__":
    main()
