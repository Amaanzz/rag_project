import os
import numpy as np
import faiss

input_folder = "../data/embeddings"

embeddings = []
file_names = []

# Load all embeddings
for file in os.listdir(input_folder):
    if file.endswith(".npy"):
        path = os.path.join(input_folder, file)

        vector = np.load(path)
        embeddings.append(vector)
        file_names.append(file)

# Convert to numpy array
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to index
index.add(embeddings)

print(f"Total vectors indexed: {index.ntotal}")

# Save index
faiss.write_index(index, "../data/faiss_index.index")

# Save mapping
np.save("../data/file_names.npy", file_names)