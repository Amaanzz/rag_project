import os
from sentence_transformers import SentenceTransformer

input_folder = "../data/chunks"
output_folder = "../data/embeddings"

os.makedirs(output_folder, exist_ok=True)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        path = os.path.join(input_folder, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Convert text → vector
        embedding = model.encode(text)

        # Save embedding
        save_path = os.path.join(output_folder, file.replace(".txt", ".npy"))

        import numpy as np
        np.save(save_path, embedding)

        print(f"Embedded: {file}")