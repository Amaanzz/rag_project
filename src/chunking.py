import os

input_folder = "../data/processed_text"
output_folder = "../data/chunks"

os.makedirs(output_folder, exist_ok=True)

CHUNK_SIZE = 500
OVERLAP = 50


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))

        start += chunk_size - overlap

    return chunks


for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        path = os.path.join(input_folder, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)

        # Save chunks
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{file.replace('.txt', '')}_chunk_{i}.txt"

            with open(os.path.join(output_folder, chunk_filename), "w", encoding="utf-8") as f:
                f.write(chunk)

        print(f"Chunked: {file} → {len(chunks)} chunks")