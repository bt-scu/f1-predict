"""
llm/retriever.py -- Semantic search over your pgvector news_chunks table.

This is the core of RAG retrieval: take a user question, embed it,
find the most semantically similar chunks in the DB, and return them
ranked by cosine similarity.
"""

from sqlalchemy import text
import pandas as pd

from db.config import engine
from llm.embeddings import F1Embedder


def retrieve_chunks(
    query: str,
    embedder: F1Embedder,
    top_k: int = 5,
    driver_filter: list[str] | None = None,
    days_back: int = 30,
) -> list[dict]:
    """
    Embed a query and find the closest chunks via pgvector cosine distance.

    Args:
        query:          Natural language question (e.g. "Will it rain at Spa?")
        embedder:       An F1Embedder instance (reuse it -- don't create per call)
        top_k:          How many chunks to return
        driver_filter:  Optional list of driver codes to narrow results, e.g. ["VER", "NOR"]
        days_back:      Only look at articles from the last N days

    Returns:
        List of dicts, each with: chunk_text, source, title, published_at,
        url, similarity (float 0-1, higher = more relevant)
    """

    # --- Step 1: Embed the query ------------------------------------------
    query_vec = embedder.embed(query)

    # --- Step 2: Build the SQL query --------------------------------------
    where_clauses = ["na.published_at >= NOW() - make_interval(days => :days)"]
    params = {"embedding": str(query_vec), "k": top_k, "days": days_back}

    if driver_filter:
        # The && operator checks array overlap:
        # '{VER,NOR}' && driver_codes is true if ANY code matches.
        where_clauses.append("nc.driver_codes && :drivers::text[]")
        params["drivers"] = "{" + ",".join(driver_filter) + "}"

    where_sql = " AND ".join(where_clauses)

    sql = text(f"""
        SELECT
            nc.chunk_id,
            nc.chunk_text,
            nc.token_count,
            na.source,
            na.title,
            na.published_at,
            na.url,
            1 - (nc.embedding <=> :embedding::vector) AS similarity
        FROM news_chunks nc
        JOIN news_articles na ON nc.article_id = na.article_id
        WHERE {where_sql}
        ORDER BY nc.embedding <=> :embedding::vector
        LIMIT :k
    """)

    # --- Step 3: Execute and return as list of dicts ----------------------
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)

    return df.to_dict("records")


if __name__ == "__main__":
    print("Loading embedder...")
    embedder = F1Embedder()

    # Change this to whatever you want to search for
    test_query = "How is Verstappen performing this season?"

    print(f"\nSearching for: '{test_query}'")
    print(f"{'='*60}")

    results = retrieve_chunks(test_query, embedder, top_k=5, days_back=365)

    if not results:
        print("\nNo results found. Either:")
        print("  1. news_chunks table is empty (need to ingest articles first)")
        print("  2. No articles match the time filter (try increasing days_back)")
    else:
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (similarity: {r['similarity']:.3f}) ---")
            print(f"Source: {r['source']} | {r['title']}")
            print(f"URL:    {r['url']}")
            print(f"Text:   {r['chunk_text'][:300]}...")