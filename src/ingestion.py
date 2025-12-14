
import boto3
import json
from pypdf import PdfReader
import numpy as np
import os

# Create Bedrock Runtime client

TITAN_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"

from dotenv import load_dotenv

load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_REGION")

bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))


def load_resume_text(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text: str, max_chars: int = 800):
    words = text.split()
    chunks = []
    current = []
    length = 0

    for w in words:
        if length + len(w) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = [w]
            length = len(w)
        else:
            current.append(w)
            length += len(w) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def get_embedding(text: str):
    body = {
        "inputText": text,
    }

    response = bedrock.invoke_model(
        modelId=TITAN_EMBEDDING_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    # According to the docs, Titan G1 returns "embedding": [float,...]
    embedding = response_body["embedding"]
    return np.array(embedding, dtype="float32")




def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Vectors must be same shape")
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)




def search_resume(query: str, top_k: int = 4):
    global resume_embeddings
    global chunks
    q_emb = get_embedding(query)
    scores = []
    for i, emb in enumerate(resume_embeddings):
        score = cosine_similarity(q_emb, emb)
        scores.append((score, i))
    scores.sort(reverse=True, key=lambda x: x[0])
    results = []
    for score, idx in scores[:top_k]:
        results.append((score, chunks[idx]))
    return results

RESUME_FILE = "./data/resume.pdf"  # change if your file name is different
full_text = load_resume_text(RESUME_FILE)
chunks = chunk_text(full_text, max_chars=800)
resume_embeddings = [get_embedding(c) for c in chunks]


if __name__ == "__main__":
    
    RESUME_FILE = "./data/resume.pdf"  # change if your file name is different

    full_text = load_resume_text(RESUME_FILE)
    chunks = chunk_text(full_text, max_chars=800)
    resume_embeddings = [get_embedding(c) for c in chunks]
    print(full_text[:100])  # show first 1000 characters to check


    # Test on a small sample:
    test_emb = get_embedding("Hello from my resume chatbot!")
    print("Embedding length:", len(test_emb))

    chunks = chunk_text(full_text, max_chars=800)
    resume_embeddings = [get_embedding(c) for c in chunks]

    print(f"Number of chunks: {len(chunks)}")
    print("Sample chunk:\n", chunks[0][:500])
    # Test search
    results = search_resume("What programming languages do I know?", top_k=3)
    for score, text in results:
        print("Score:", round(score, 3))
        print(text[:200], "\n---\n")
