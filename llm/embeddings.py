"""
llm/embeddings.py -- Embedding wrapper for the RAG pipeline.

WHAT THIS FILE DOES:
    Converts text (article chunks, user queries) into 384-dimensional vectors
    using the all-MiniLM-L6-v2 model. These vectors capture semantic meaning,
    so "Red Bull improved downforce" and "RB21 gains aerodynamic load" produce
    similar vectors even though they share few keywords.

WHY A WRAPPER:
    - Load the model ONCE and reuse it (loading takes ~2 seconds, encoding is fast)
    - Single place to swap models later (e.g., upgrade to a larger model)
    - Batch encoding for performance (embed 100 chunks at once, not one at a time)

USAGE:
    from llm.embeddings import F1Embedder

    embedder = F1Embedder()
    vec = embedder.embed("Verstappen fastest in FP2")              # single text
    vecs = embedder.embed_batch(["chunk 1 text", "chunk 2 text"])  # many at once
"""

from sentence_transformers import SentenceTransformer
import numpy as np


# --------------------------------------------------------------------------- #
# Config -- change these if you swap models later
# --------------------------------------------------------------------------- #
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, ~80MB, runs fine on CPU
EMBEDDING_DIM = 384               # must match vector(384) in your DB schema


class F1Embedder:
    """
    Thin wrapper around sentence-transformers.

    TIP: This class should be instantiated ONCE (it's expensive to load the model).
         Pass the instance around or store it at module level.
         Do NOT create a new F1Embedder() for every embed call.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        # TODO: Load the SentenceTransformer model here.
        #
        # This is where the ~80MB model gets downloaded (first run only)
        # and loaded into memory. After the first download it's cached
        # in ~/.cache/huggingface/
        #
        # Hint: self.model = SentenceTransformer(model_name)
        pass

    def embed(self, text: str) -> list[float]:
        """
        Embed a single piece of text and return a list of floats.

        This is what you'll call for:
          - Embedding a user's question at query time
          - Embedding a single chunk during ingestion

        TODO:
          1. Use self.model.encode(text) to get a numpy array
          2. Convert to a plain Python list with .tolist()
          3. Return it

        TIP: .encode() returns a numpy array of shape (384,).
             pgvector expects a plain Python list, so always .tolist() it
             before inserting into the DB.

        TIP: For query-time embedding, this function is all you need.
             It takes ~5ms per call on CPU -- plenty fast for interactive use.
        """
        pass

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Embed multiple texts at once. Much faster than calling embed() in a loop.

        This is what you'll call during ingestion when you have many chunks
        from one or more articles to embed at the same time.

        Args:
            texts:      List of chunk texts to embed
            batch_size: How many to encode at once (64 is a good default for CPU).
                        The model processes this many in parallel using vectorized ops.

        Returns:
            List of embedding vectors (each is a list of 384 floats)

        TODO:
        
          1. Use self.model.encode(texts, batch_size=batch_size)
             - This returns a numpy array of shape (len(texts), 384)
          2. Convert to list of lists with .tolist()
          3. Return it

        TIP: On CPU, all-MiniLM-L6-v2 can embed ~100-200 chunks/second in batch.
             A typical article produces ~8 chunks, so batch embedding a whole
             article takes <0.1 seconds.
        """
        pass


# --------------------------------------------------------------------------- #
# Helper functions you may want later
# --------------------------------------------------------------------------- #

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors. Returns a value between -1 and 1.
    1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite.

    This is useful for:
      - Deduplication: if similarity > 0.95, the articles are near-duplicates
      - Debugging: check if your embeddings make sense by comparing known-similar texts

    TODO:
      1. Convert both to numpy arrays
      2. Use the formula: dot(a, b) / (norm(a) * norm(b))
      3. Return the float

    TIP: You won't use this for search (pgvector does that in SQL with the <=> operator).
         This is a utility for testing and dedup logic in Python.
    """
    pass
