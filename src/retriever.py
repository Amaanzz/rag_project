import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index
index = faiss.read_index("../data/faiss_index.index")

# Load file names mapping
file_names = np.load("../data/file_names.npy", allow_pickle=True)


def retrieve(query, top_k=5):
    # Convert query → embedding
    query_vector = model.encode([query]).astype("float32")

    # Search in FAISS
    distances, indices = index.search(query_vector, top_k)

    results = []

    for idx in indices[0]:
        file_name = file_names[idx]

        # ✅ Store result (IMPORTANT FIX)
        results.append(file_name)

        # Convert embedding filename → chunk filename
        chunk_file = file_name.replace(".npy", ".txt")
        chunk_path = os.path.join("../data/chunks", chunk_file)

        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                content = f.read()

            print("\n--- Retrieved Chunk ---")
            print(content[:300])  # show first 300 characters

        except Exception as e:
            print(f"Error reading {chunk_file}: {e}")

    return results


# ===========================
# TEST QUERY
# ===========================

query = "What is retrieval augmented generation?"

results = retrieve(query)

print("\nTop Results:")
for r in results:
    print(r)