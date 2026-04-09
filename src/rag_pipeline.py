import faiss
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama
from hallucination import detect_hallucination

# =========================
# LOAD MODELS
# =========================

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# =========================
# LOAD DATABASE
# =========================

index = faiss.read_index("../data/faiss_index.index")
file_names = np.load("../data/file_names.npy", allow_pickle=True)

# =========================
# RETRIEVE CONTEXT (TOP 50)
# =========================

def retrieve_context(query, top_k=50):
    query_vector = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    chunks = []
    sources = []

    for idx in indices[0]:
        file_name = file_names[idx]
        chunk_file = file_name.replace(".npy", ".txt")
        chunk_path = os.path.join("../data/chunks", chunk_file)

        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                content = f.read()

            if content.strip():
                chunks.append(content)
                sources.append(chunk_file)

        except Exception as e:
            print(f"Error reading {chunk_file}: {e}")

    return chunks, sources


# =========================
# CROSS-ENCODER RE-RANKING
# =========================

def rerank_chunks(query, chunks, sources, top_k=3):
    pairs = [(query, chunk) for chunk in chunks]

    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(chunks, sources, scores),
        key=lambda x: x[2],
        reverse=True
    )

    top_chunks = [item[0] for item in ranked[:top_k]]
    top_sources = [item[1] for item in ranked[:top_k]]

    return top_chunks, top_sources


# =========================
# GENERATION (MISTRAL)
# =========================

def generate_text(prompt):
    response = ollama.chat(
        model='mistral',
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']


# =========================
# MAIN RAG FUNCTION
# =========================

def generate_answer(query):
    # Step 1: Retrieve
    chunks, sources = retrieve_context(query)

    # Step 2: Re-rank (top 3)
    top_chunks, top_sources = rerank_chunks(query, chunks, sources)

    print("\n[DEBUG] Top Sources After Re-ranking:")
    for i, src in enumerate(top_sources):
        print(f"{i+1}. {src}")

    # =========================
    # SMART CONTEXT BUILDING
    # =========================

    context_list = []

    for chunk in top_chunks:
        clean_chunk = re.sub(r"\[\d+\]", "", chunk)
        context_list.append(clean_chunk[:400])

    context = "\n\n".join(context_list)

    print("\n[DEBUG] Context Length:", len(context))
    print("\n[DEBUG] Context Preview:\n", context[:300])

    # =========================
    # FINAL PROMPT
    # =========================

    sources_str = ", ".join(top_sources)

    prompt = f"""
You are an expert AI researcher. 
Answer the question using ONLY the provided Context. If you do not know the answer based on the Context, say "I cannot find this in the provided documents."

Context:
{context}

Question:
{query}

Instructions:
1. Provide the answer in exactly 2–3 short sentences, each on a new line.
2. **Bold** the core concepts or definitions (e.g., **Retrieval-Augmented Generation**).
3. Do not invent acronyms. Rely strictly on the facts in the Context.
4. Conclude with: "Source: {sources_str}"

Answer:
"""

    answer = generate_text(prompt)

    return answer, top_sources, context


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    query = "What is retrieval augmented generation?"

    answer, sources, context = generate_answer(query)

    print("\n================ ANSWER ================\n")
    print(answer)

    print("\n============= SOURCES USED =============\n")
    for s in sources:
        print(s)

    # =========================
    # HALLUCINATION CHECK
    # =========================

    score, status = detect_hallucination(answer, context)

    print("\n============= HALLUCINATION CHECK =============\n")
    print(f"Similarity Score: {score:.2f}")
    print(f"Status: {status}")