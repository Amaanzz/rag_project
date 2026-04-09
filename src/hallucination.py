from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


def compute_similarity(text1, text2):
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)

    similarity = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    return similarity


def detect_hallucination(answer, context):
    score = compute_similarity(answer, context)

    if score > 0.6:
        status = "LOW hallucination risk ✅"
    elif score > 0.4:
        status = "MEDIUM hallucination risk ⚠️"
    else:
        status = "HIGH hallucination risk ❌"

    return score, status