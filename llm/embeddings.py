"""
llm/embeddings.py -- Embedding wrapper for the RAG pipeline.

Converts text (article chunks, user queries) into 384-dimensional vectors
using the all-MiniLM-L6-v2 model. These vectors capture semantic meaning,
so "Red Bull improved downforce" and "RB21 gains aerodynamic load" produce
similar vectors even though they share few keywords.

USAGE:
    from llm.embeddings import F1Embedder

    embedder = F1Embedder()
    vec = embedder.embed("Verstappen fastest in FP2")              # single text
    vecs = embedder.embed_batch(["chunk 1 text", "chunk 2 text"])  # many at once
"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384            


class F1Embedder:
    """
    Thin wrapper around sentence-transformers.

    Instantiate ONCE (loading the model takes ~2 seconds).
    Pass the instance around -- don't create a new one per call.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text and return a list of 384 floats."""
        np_arr = self.model.encode(text)
        return np_arr.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed multiple texts at once. Returns list of 384-float lists."""
        np_arr = self.model.encode(texts, batch_size=batch_size)
        return np_arr.tolist()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Cosine similarity between two vectors. Returns -1.0 to 1.0.
    1.0 = identical meaning, 0.0 = unrelated.

    Useful for dedup (> 0.95 = near-duplicate) and debugging embeddings.
    Not needed for search -- pgvector does that in SQL with the <=> operator.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))