Your Current State (Gap Analysis)                                                                                        
  ┌────────────────┬──────────┬──────────────────────────────────────────────────┐
  │     Layer      │  Status  │                   What Exists                    │
  ├────────────────┼──────────┼──────────────────────────────────────────────────┤
  │ Data Ingestion │ Partial  │ FastF1 API + CSV/JSON, but no news/articles      │
  ├────────────────┼──────────┼──────────────────────────────────────────────────┤
  │ ML Models      │ Strong   │ XGBoost quali/sprint/race with SHAP + risk flags │
  ├────────────────┼──────────┼──────────────────────────────────────────────────┤
  │ Database       │ Strong   │ PostgreSQL 13 tables, but no pgvector            │
  ├────────────────┼──────────┼──────────────────────────────────────────────────┤
  │ LLM            │ Skeleton │ llm/basic.py = Gemini hello-world only           │
  ├────────────────┼──────────┼──────────────────────────────────────────────────┤
  │ Vector Search  │ None     │ No embeddings, no retriever                      │
  ├────────────────┼──────────┼──────────────────────────────────────────────────┤
  │ API Layer      │ None     │ No REST endpoints                                │
  ├────────────────┼──────────┼──────────────────────────────────────────────────┤
  │ Scheduler      │ None     │ No automated updates                             │
  └────────────────┴──────────┴──────────────────────────────────────────────────┘
  ---
  1. RAG Pipeline Architecture

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        INGESTION LAYER                                  │
  │                                                                         │
  │  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐              │
  │  │ RSS Feeds│  │ YouTube  │  │ Reddit    │  │ FastF1   │              │
  │  │ (F1.com, │  │Transcripts│ │ r/formula1│  │ API      │              │
  │  │ RaceFans,│  │(tech talk)│  │(comments) │  │(existing)│              │
  │  │ TheRace) │  └────┬─────┘  └─────┬─────┘  └────┬─────┘              │
  │  └────┬─────┘       │              │              │                     │
  │       └─────────────┴──────┬───────┴──────────────┘                     │
  │                            ▼                                            │
  │                   ┌─────────────────┐                                   │
  │                   │  Scheduler       │                                   │
  │                   │  (APScheduler)   │                                   │
  │                   └────────┬────────┘                                   │
  └────────────────────────────┼────────────────────────────────────────────┘
                               ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                      PROCESSING LAYER                                   │
  │                                                                         │
  │  ┌────────────┐   ┌──────────────┐   ┌────────────────┐                │
  │  │ Clean &    │──▶│ Semantic     │──▶│ Embed          │                │
  │  │ Normalize  │   │ Chunk        │   │ (all-MiniLM    │                │
  │  │ (BS4,      │   │ (spaCy NLP   │   │  -L6-v2, 384d) │                │
  │  │  trafilatura)│  │  boundaries) │   └───────┬────────┘                │
  │  └────────────┘   └──────────────┘           │                         │
  │                                               ▼                         │
  │                                    ┌──────────────────┐                 │
  │                          ┌─────── │ Metadata Extract  │                 │
  │                          │        │ (driver mentions,  │                │
  │                          │        │  circuit, date,    │                │
  │                          │        │  topic tags)       │                │
  │                          │        └──────────────────┘                  │
  └──────────────────────────┼──────────────────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                       STORAGE LAYER                                     │
  │                                                                         │
  │  ┌────────────────────────────────────────────────────┐                 │
  │  │              PostgreSQL (localhost:6000)            │                 │
  │  │                                                    │                 │
  │  │  EXISTING:           │  NEW (RAG):                 │                 │
  │  │  races               │  news_articles              │                 │
  │  │  results             │  news_chunks                │                 │
  │  │  qualifying          │  chunk_embeddings (pgvector)│                 │
  │  │  practice_laps       │  ingestion_log              │                 │
  │  │  driver_predictions  │  retrieval_feedback         │                 │
  │  │  driver_fantasy_*    │                             │                 │
  │  └────────────────────────────────────────────────────┘                 │
  └─────────────────────────────────────────────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                      RETRIEVAL LAYER                                    │
  │                                                                         │
  │  User Query ──▶ ┌──────────────┐    ┌───────────────────┐              │
  │                 │ Query Embed  │──▶ │ Hybrid Search      │              │
  │                 │ (same model) │    │ (pgvector cosine   │              │
  │                 └──────────────┘    │  + BM25 keyword    │              │
  │                                     │  + metadata filter)│              │
  │                                     └────────┬──────────┘              │
  │                                              ▼                          │
  │                                     ┌───────────────────┐              │
  │                                     │ Re-rank (cross-   │              │
  │                                     │  encoder or        │              │
  │                                     │  Cohere reranker)  │              │
  │                                     └────────┬──────────┘              │
  └──────────────────────────────────────────────┼──────────────────────────┘
                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                    GENERATION LAYER                                      │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────┐       │
  │  │                  Context Builder                             │       │
  │  │                                                              │       │
  │  │  ML Predictions (from driver_predictions)                    │       │
  │  │  + Retrieved News Chunks (top-k)                             │       │
  │  │  + Fantasy Rules (budget, pricing)                           │       │
  │  │  + Weather Forecast                                          │       │
  │  │                        ▼                                     │       │
  │  │              ┌──────────────────┐                            │       │
  │  │              │  Prompt Template │                            │       │
  │  │              └────────┬─────────┘                            │       │
  │  │                       ▼                                      │       │
  │  │    ┌──────────────────────────────────────┐                  │       │
  │  │    │  LLM (Gemini 2.5 Flash / Claude)     │                  │       │
  │  │    └──────────────────┬───────────────────┘                  │       │
  │  └───────────────────────┼──────────────────────────────────────┘       │
  │                          ▼                                              │
  │                   Final Response                                        │
  └─────────────────────────────────────────────────────────────────────────┘

  ---
  2. F1 News Sources & Ingestion Strategy

  Tier 1: Reliable, Free, RSS-based (Start Here)
  Source: The Race
  Method: RSS (therace.com/feed)
  Content Type: Technical analysis, long-form
  Update Freq: 5-10/day
  ────────────────────────────────────────
  Source: RaceFans
  Method: RSS (racefans.net/feed)
  Content Type: Technical, race reviews
  Update Freq: 3-8/day
  ────────────────────────────────────────
  Source: Motorsport.com
  Method: RSS
  Content Type: Breaking news, tech
  Update Freq: 10-20/day
  ────────────────────────────────────────
  Source: Autosport
  Method: RSS
  Content Type: Technical features
  Update Freq: 5-15/day
  ────────────────────────────────────────
  Source: Formula1.com
  Method: RSS (formula1.com/content/fom-website/en/latest/all.xml)
  Content Type: Official news
  Update Freq: 3-5/day
  ────────────────────────────────────────
  Source: r/formula1
  Method: Reddit API (free tier)
  Content Type: Community analysis, rumors
  Update Freq: Continuous
  Tier 2: Deeper Content (Phase 2)
  ┌─────────────────────────┬─────────────────────────────────────────────┬─────────────────────────────────────────┐
  │         Source          │                   Method                    │              Content Type               │
  ├─────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ YouTube (tech channels) │ yt-dlp + whisper transcription              │ Driver's Eye, Chain Bear, Peter Windsor │
  ├─────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ FIA Documents           │ PDF scrape (regulations, steward decisions) │ Technical directives, penalties         │
  ├─────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ Twitter/X               │ API (paid, ~$100/mo Basic)                  │ Real-time paddock updates               │
  └─────────────────────────┴─────────────────────────────────────────────┴─────────────────────────────────────────┘
  Tier 3: Premium (Phase 3)
  ┌──────────────────┬─────────────────────┬────────────┐
  │      Source      │       Method        │    Cost    │
  ├──────────────────┼─────────────────────┼────────────┤
  │ F1 TV API        │ Official API access │ ~$80/year  │
  ├──────────────────┼─────────────────────┼────────────┤
  │ Motorsport Stats │ Licensed data API   │ ~$500/year │
  └──────────────────┴─────────────────────┴────────────┘
  Licensing & Compliance

  # Always include in your scraper:
  RESPECTFUL_SCRAPING = {
      "User-Agent": "F1PredictBot/1.0 (personal research; contact@youremail.com)",
      "rate_limit": 1,          # 1 request per second max
      "robots_txt": True,       # Always check robots.txt first
      "cache_hours": 6,         # Don't re-fetch same article within 6 hours
      "no_paywalled": True,     # Skip paywalled content
  }

  - RSS feeds are explicitly intended for programmatic consumption -- always safe
  - Reddit API has clear terms (free tier: 100 req/min, must identify as bot)
  - YouTube transcription of public content falls under fair use for personal analysis
  - Never scrape paywalled content (The Athletic, F1 TV exclusive articles)

  Novelty Detection

  # Deduplication strategy
  def is_novel(article_url: str, article_text: str, engine) -> bool:
      # 1. URL dedup (exact match)
      exists = pd.read_sql(
          "SELECT 1 FROM news_articles WHERE url = :url", engine, params={"url": article_url}
      )
      if not exists.empty:
          return False

      # 2. Content similarity dedup (catch reposts/syndication)
      embedding = embed(article_text)
      similar = vector_search(embedding, threshold=0.95, limit=1)
      if similar:
          return False  # Near-duplicate exists

      return True

  ---
  3. Database Schema (New Tables for RAG)

  Add these tables to your existing PostgreSQL. Use pgvector since you already run PostgreSQL -- no need for a separate
  vector DB.

  -- Enable pgvector extension (run once)
  CREATE EXTENSION IF NOT EXISTS vector;

  -- Raw articles/content
  CREATE TABLE news_articles (
      article_id    SERIAL PRIMARY KEY,
      source        TEXT NOT NULL,                -- 'therace', 'racefans', 'reddit', 'youtube'
      url           TEXT UNIQUE NOT NULL,
      title         TEXT,
      author        TEXT,
      content       TEXT NOT NULL,                -- full cleaned text
      published_at  TIMESTAMP WITH TIME ZONE,
      ingested_at   TIMESTAMP DEFAULT NOW(),
      driver_codes  TEXT[],                       -- '{VER,HAM,NOR}'
      circuit_name  TEXT,                         -- links to races.circuit_name
      content_type  TEXT DEFAULT 'article',       -- 'article','transcript','discussion','regulation'
      word_count    INTEGER,
      content_hash  TEXT UNIQUE                   -- SHA256 for dedup
  );

  -- Chunked content with embeddings
  CREATE TABLE news_chunks (
      chunk_id      SERIAL PRIMARY KEY,
      article_id    INTEGER REFERENCES news_articles(article_id) ON DELETE CASCADE,
      chunk_index   INTEGER NOT NULL,             -- order within article
      chunk_text    TEXT NOT NULL,
      token_count   INTEGER,
      embedding     vector(384),                  -- all-MiniLM-L6-v2 output
      driver_codes  TEXT[],                       -- inherited or chunk-specific
      created_at    TIMESTAMP DEFAULT NOW()
  );

  -- HNSW index for fast approximate nearest neighbor search
  CREATE INDEX idx_chunks_embedding ON news_chunks
      USING hnsw (embedding vector_cosine_ops)
      WITH (m = 16, ef_construction = 200);

  -- For metadata-filtered searches
  CREATE INDEX idx_chunks_drivers ON news_chunks USING gin (driver_codes);
  CREATE INDEX idx_chunks_article ON news_chunks (article_id);

  -- Ingestion tracking
  CREATE TABLE ingestion_log (
      log_id        SERIAL PRIMARY KEY,
      source        TEXT NOT NULL,
      run_at        TIMESTAMP DEFAULT NOW(),
      articles_found  INTEGER DEFAULT 0,
      articles_new    INTEGER DEFAULT 0,
      chunks_created  INTEGER DEFAULT 0,
      errors          JSONB,
      duration_sec    FLOAT
  );

  -- Retrieval quality feedback (for evaluation)
  CREATE TABLE retrieval_feedback (
      feedback_id   SERIAL PRIMARY KEY,
      query_text    TEXT,
      query_embedding vector(384),
      retrieved_chunk_ids INTEGER[],
      llm_response  TEXT,
      user_rating   INTEGER CHECK (user_rating BETWEEN 1 AND 5),
      created_at    TIMESTAMP DEFAULT NOW()
  );

  Why pgvector Over Pinecone/Chroma
  ┌────────────────────────┬───────────────────────────────────┬──────────────────────────┬───────────────────────┐
  │         Factor         │             pgvector              │         Pinecone         │        Chroma         │
  ├────────────────────────┼───────────────────────────────────┼──────────────────────────┼───────────────────────┤
  │ Your case              │ You already have PostgreSQL       │ New service to manage    │ Separate process      │
  ├────────────────────────┼───────────────────────────────────┼──────────────────────────┼───────────────────────┤
  │ Cost                   │ Free                              │ $70/mo+                  │ Free but memory-heavy │
  ├────────────────────────┼───────────────────────────────────┼──────────────────────────┼───────────────────────┤
  │ JOINs with your data   │ Native SQL JOINs to races/drivers │ Requires app-level joins │ Separate query        │
  ├────────────────────────┼───────────────────────────────────┼──────────────────────────┼───────────────────────┤
  │ Scaling to 100K chunks │ Excellent with HNSW               │ Excellent                │ Good                  │
  ├────────────────────────┼───────────────────────────────────┼──────────────────────────┼───────────────────────┤
  │ Scaling to 10M+ chunks │ Needs tuning                      │ Excellent                │ Poor                  │
  └────────────────────────┴───────────────────────────────────┴──────────────────────────┴───────────────────────┘
  For your scale (F1 news = likely 10K-100K chunks over years), pgvector is the right call. You can JOIN retrieval results
  directly with driver_predictions and races tables in a single query.

  ---
  4. Semantic Chunking Strategy for F1 Technical Text

  F1 technical articles have specific structure patterns. Generic chunking will lose context.

  import spacy
  from sentence_transformers import SentenceTransformer

  nlp = spacy.load("en_core_web_sm")
  embedder = SentenceTransformer("all-MiniLM-L6-v2")

  def semantic_chunk_f1_article(text: str, max_tokens: int = 256, overlap_sentences: int = 2) -> list[dict]:
      """
      Chunk F1 technical articles using semantic boundaries.

      Strategy:
      1. Split into sentences via spaCy
      2. Group sentences into candidate chunks (~256 tokens)
      3. Detect topic shifts via embedding similarity between consecutive chunks
      4. Split at topic shifts; merge small fragments
      5. Add overlap (last 2 sentences of previous chunk prepended)
      """
      doc = nlp(text)
      sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

      chunks = []
      current_chunk = []
      current_tokens = 0

      for sent in sentences:
          sent_tokens = len(sent.split())  # rough token estimate
          if current_tokens + sent_tokens > max_tokens and current_chunk:
              chunks.append({
                  "text": " ".join(current_chunk),
                  "token_count": current_tokens,
              })
              # Overlap: carry last N sentences forward
              current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences else []
              current_tokens = sum(len(s.split()) for s in current_chunk)

          current_chunk.append(sent)
          current_tokens += sent_tokens

      if current_chunk:
          chunks.append({"text": " ".join(current_chunk), "token_count": current_tokens})

      return chunks

  F1-Specific Chunking Rules

  1. Never split mid-paragraph about a single car/driver -- "Red Bull's RB21 features a new floor edge..." should stay
  together
  2. Keep regulation references with their context -- "Under Art. 3.5.1..." needs the explanation that follows
  3. Preserve comparison blocks -- "McLaren's sidepod vs Ferrari's approach" needs both sides
  4. Target 200-300 tokens per chunk -- F1 tech articles are information-dense; smaller chunks lose context
  5. 20% overlap (2 sentences) -- prevents losing cross-sentence references like "This modification..."

  When to Re-chunk

  - Never re-chunk existing articles unless your embedding model changes
  - Version your embeddings -- store model name in metadata

  ---
  5. LLM Integration: Retrieval + Prompt Templates

  Context Builder

  This combines your ML predictions with retrieved news into a single prompt.

  # llm/context_builder.py

  def build_race_analysis_context(
      race_id: int,
      query: str,
      engine,
      embedder,
      top_k: int = 8,
      max_context_tokens: int = 6000
  ) -> str:
      """
      Assemble context from:
      1. ML predictions (driver_predictions table)
      2. Retrieved news chunks (semantic search)
      3. Weather forecast
      4. Fantasy pricing
      """

      # 1. ML Predictions
      predictions = pd.read_sql(text("""
          SELECT d.full_name, d.team_name,
                 dp.pred_quali_pos, dp.pred_race_pos,
                 dp.pred_race_pos_p10, dp.pred_race_pos_p90,
                 dp.prob_race_dnf, dp.ev_race_points,
                 dp.pred_race_overtakes, dp.race_factors_json,
                 dp.risk_flags_json
          FROM driver_predictions dp
          JOIN drivers d ON dp.driver_id = d.driver_id
          WHERE dp.race_id = :race_id
          ORDER BY dp.pred_race_pos
      """), engine, params={"race_id": race_id})

      pred_context = format_predictions_table(predictions)

      # 2. Semantic search for relevant news
      query_embedding = embedder.encode(query)
      news_chunks = retrieve_chunks(
          query_embedding=query_embedding,
          engine=engine,
          top_k=top_k,
          race_id=race_id  # optional: filter by circuit
      )

      news_context = "\n\n".join([
          f"[{c['source']} | {c['published_at']:%Y-%m-%d}]\n{c['chunk_text']}"
          for c in news_chunks
      ])

      # 3. Weather
      weather = pd.read_sql(text("""
          SELECT air_temp, track_temp, humidity, rainfall, wind_speed
          FROM weather WHERE race_id = :race_id AND session_type = 'race'
          LIMIT 1
      """), engine, params={"race_id": race_id})

      return f"""## ML Model Predictions
  {pred_context}

  ## Recent F1 News & Analysis
  {news_context}

  ## Race Weather
  {weather.to_string(index=False) if not weather.empty else "Forecast unavailable"}
  """

  Prompt Template

  # llm/prompts/race_analysis.py

  RACE_ANALYSIS_PROMPT = """You are an expert Formula 1 analyst with deep knowledge of
  car engineering, race strategy, tire management, and driver performance.

  You have access to:
  1. XGBoost ML model predictions with SHAP feature importance
  2. Recent technical news and analysis articles
  3. Weather data for the upcoming race

  CONTEXT:
  {context}

  USER QUESTION:
  {query}

  INSTRUCTIONS:
  - Ground your analysis in the ML predictions AND the news context provided
  - When citing predictions, mention confidence intervals (p10/p90 range)
  - If news contradicts model predictions (e.g., car upgrade not in model), flag it explicitly
  - Mention risk flags (rain sensitivity, DNF proneness) when relevant
  - Be specific about which sources inform your answer
  - If information is insufficient, say so rather than speculate

  Provide your analysis:"""

  Handling Context Window Limits

  def fit_to_context_window(
      ml_context: str,
      news_chunks: list[str],
      max_tokens: int = 8000  # Gemini Flash = 1M, but shorter = better answers
  ) -> str:
      """
      Priority order for context stuffing:
      1. ML predictions (always include -- ~500 tokens)
      2. Weather (always include -- ~50 tokens)
      3. News chunks (fill remaining budget, most relevant first)
      """
      budget = max_tokens - count_tokens(ml_context) - 500  # reserve for prompt template

      included_news = []
      for chunk in news_chunks:  # already ranked by relevance
          chunk_tokens = count_tokens(chunk)
          if budget - chunk_tokens < 0:
              break
          included_news.append(chunk)
          budget -= chunk_tokens

      return ml_context + "\n\n".join(included_news)

  Multi-LLM Support

  # llm/config.py

  LLM_CONFIGS = {
      "gemini-flash": {
          "provider": "google",
          "model": "gemini-2.5-flash",
          "max_context": 1_000_000,
          "cost_per_1k_input": 0.0,      # free tier
          "daily_limit": 1_500_000,
          "best_for": "fast analysis, high volume"
      },
      "claude-sonnet": {
          "provider": "anthropic",
          "model": "claude-sonnet-4-5-20250929",
          "max_context": 200_000,
          "cost_per_1k_input": 0.003,
          "best_for": "deep reasoning, nuanced analysis"
      },
      "claude-haiku": {
          "provider": "anthropic",
          "model": "claude-haiku-4-5-20251001",
          "max_context": 200_000,
          "cost_per_1k_input": 0.0008,
          "best_for": "fast, cheap classification tasks"
      }
  }

  ---
  6. Continuous Updating Workflow

  # llm/scheduler.py

  from apscheduler.schedulers.background import BackgroundScheduler
  from apscheduler.triggers.cron import CronTrigger

  scheduler = BackgroundScheduler()

  # RSS feeds: every 2 hours during race weekends, every 6 hours otherwise
  scheduler.add_job(
      ingest_rss_feeds,
      CronTrigger(hour="*/2"),  # adjust dynamically for race weekends
      id="rss_ingestion",
      kwargs={"sources": ["therace", "racefans", "motorsport", "f1.com"]}
  )

  # Reddit: every 4 hours (rate limit friendly)
  scheduler.add_job(
      ingest_reddit,
      CronTrigger(hour="*/4"),
      id="reddit_ingestion",
      kwargs={"subreddit": "formula1", "min_score": 50}
  )

  # YouTube transcripts: daily at 6 AM UTC
  scheduler.add_job(
      ingest_youtube_transcripts,
      CronTrigger(hour=6),
      id="youtube_ingestion",
      kwargs={"channels": ["chainbear", "peterwindsor", "f1official"]}
  )

  # ML model re-prediction: Friday evening after FP2
  scheduler.add_job(
      run_all_predictions,
      CronTrigger(day_of_week="fri", hour=20),
      id="ml_predictions"
  )

  Incremental Vector DB Updates (No Full Re-embed)

  def ingest_new_articles(source: str, articles: list[dict], engine, embedder):
      """Only embed and store NEW content. Never re-embed existing."""
      new_count = 0

      for article in articles:
          content_hash = hashlib.sha256(article["content"].encode()).hexdigest()

          # Skip if already exists (by URL or content hash)
          exists = engine.execute(text(
              "SELECT 1 FROM news_articles WHERE url = :url OR content_hash = :hash"
          ), {"url": article["url"], "hash": content_hash}).fetchone()

          if exists:
              continue

          # Insert article
          result = engine.execute(text("""
              INSERT INTO news_articles (source, url, title, author, content, published_at,
                                         driver_codes, circuit_name, content_hash, word_count)
              VALUES (:source, :url, :title, :author, :content, :published_at,
                      :driver_codes, :circuit_name, :hash, :word_count)
              RETURNING article_id
          """), {
              "source": source, "url": article["url"], "title": article["title"],
              "author": article.get("author"), "content": article["content"],
              "published_at": article["published_at"],
              "driver_codes": extract_driver_codes(article["content"]),
              "circuit_name": extract_circuit(article["content"]),
              "hash": content_hash,
              "word_count": len(article["content"].split())
          })
          article_id = result.fetchone()[0]

          # Chunk and embed
          chunks = semantic_chunk_f1_article(article["content"])
          for i, chunk in enumerate(chunks):
              embedding = embedder.encode(chunk["text"]).tolist()
              engine.execute(text("""
                  INSERT INTO news_chunks (article_id, chunk_index, chunk_text, token_count,
                                           embedding, driver_codes)
                  VALUES (:aid, :idx, :text, :tokens, :emb, :drivers)
              """), {
                  "aid": article_id, "idx": i, "text": chunk["text"],
                  "tokens": chunk["token_count"], "emb": str(embedding),
                  "drivers": extract_driver_codes(chunk["text"])
              })

          new_count += 1

      return new_count

  ---
  7. Retrieval Function (Hybrid Search)

  # llm/retriever.py

  def retrieve_chunks(
      query_embedding: list[float],
      engine,
      top_k: int = 10,
      driver_filter: list[str] = None,
      days_back: int = 30,
      keyword_query: str = None
  ) -> list[dict]:
      """
      Hybrid retrieval: pgvector cosine similarity + optional metadata filters.
      """
      filters = ["na.published_at >= NOW() - INTERVAL ':days days'"]
      params = {"embedding": str(query_embedding), "k": top_k, "days": days_back}

      if driver_filter:
          filters.append("nc.driver_codes && :drivers")  # array overlap
          params["drivers"] = driver_filter

      where_clause = " AND ".join(filters)

      query = text(f"""
          SELECT nc.chunk_id, nc.chunk_text, nc.token_count,
                 na.source, na.title, na.published_at, na.url,
                 1 - (nc.embedding <=> :embedding::vector) as similarity
          FROM news_chunks nc
          JOIN news_articles na ON nc.article_id = na.article_id
          WHERE {where_clause}
          ORDER BY nc.embedding <=> :embedding::vector
          LIMIT :k
      """)

      results = pd.read_sql(query, engine, params=params)
      return results.to_dict('records')

  ---
  8. Evaluation Metrics & Logging

  Retrieval Quality

  # llm/evaluation.py

  def evaluate_retrieval(test_queries: list[dict], engine, embedder) -> dict:
      """
      Evaluate retrieval quality on a labeled test set.

      test_queries format:
      [{"query": "RB21 floor changes", "relevant_article_ids": [12, 45, 67]}]
      """
      metrics = {"precision_at_5": [], "recall_at_5": [], "mrr": [], "ndcg_at_10": []}

      for tq in test_queries:
          query_emb = embedder.encode(tq["query"])
          retrieved = retrieve_chunks(query_emb, engine, top_k=10)
          retrieved_article_ids = [r["article_id"] for r in retrieved]
          relevant = set(tq["relevant_article_ids"])

          # Precision@5
          top5 = retrieved_article_ids[:5]
          p5 = len(set(top5) & relevant) / 5
          metrics["precision_at_5"].append(p5)

          # Recall@5
          r5 = len(set(top5) & relevant) / len(relevant) if relevant else 0
          metrics["recall_at_5"].append(r5)

          # MRR (Mean Reciprocal Rank)
          for rank, aid in enumerate(retrieved_article_ids, 1):
              if aid in relevant:
                  metrics["mrr"].append(1.0 / rank)
                  break
          else:
              metrics["mrr"].append(0.0)

      return {k: sum(v)/len(v) for k, v in metrics.items()}

  Pipeline Performance Dashboard
  ┌───────────────────────────┬───────────────────┬───────────────────────────────────────────┐
  │          Metric           │      Target       │              How to Measure               │
  ├───────────────────────────┼───────────────────┼───────────────────────────────────────────┤
  │ Retrieval precision@5     │ > 0.7             │ Labeled test set (build 50+ queries)      │
  ├───────────────────────────┼───────────────────┼───────────────────────────────────────────┤
  │ Retrieval recall@5        │ > 0.6             │ Same test set                             │
  ├───────────────────────────┼───────────────────┼───────────────────────────────────────────┤
  │ MRR                       │ > 0.5             │ First relevant result in top 2 on average │
  ├───────────────────────────┼───────────────────┼───────────────────────────────────────────┤
  │ Ingestion latency         │ < 30s per article │ ingestion_log.duration_sec                │
  ├───────────────────────────┼───────────────────┼───────────────────────────────────────────┤
  │ Embedding throughput      │ > 100 chunks/sec  │ all-MiniLM-L6-v2 on CPU                   │
  ├───────────────────────────┼───────────────────┼───────────────────────────────────────────┤
  │ Query-to-response latency │ < 3s total        │ End-to-end (embed + search + LLM)         │
  ├───────────────────────────┼───────────────────┼───────────────────────────────────────────┤
  │ Storage growth            │ ~1GB/year         │ ~50K chunks x 384 floats x 4 bytes        │
  ├───────────────────────────┼───────────────────┼───────────────────────────────────────────┤
  │ LLM answer factuality     │ Manual review     │ Spot-check 10% of responses weekly        │
  ├───────────────────────────┼───────────────────┼───────────────────────────────────────────┤
  │ Cost/month                │ < $5              │ Gemini free tier + local embeddings       │
  └───────────────────────────┴───────────────────┴───────────────────────────────────────────┘
  Logging

  # Log every retrieval for future evaluation
  def log_retrieval(query, retrieved_chunks, llm_response, engine):
      engine.execute(text("""
          INSERT INTO retrieval_feedback
              (query_text, retrieved_chunk_ids, llm_response)
          VALUES (:query, :chunk_ids, :response)
      """), {
          "query": query,
          "chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
          "response": llm_response
      })

  ---
  9. Security & Access Control
  ┌──────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
  │         Concern          │                                         Solution                                         │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ API keys exposed         │ Already using .env + .gitignore -- good. Move to python-decouple or vault for production │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ DB credentials hardcoded │ Currently f1-pass is in every .py file. Centralize to single config.py reading from .env │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Vector DB data leakage   │ pgvector lives in your PostgreSQL -- same access controls as existing tables             │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Who can query            │ Add API key auth to FastAPI endpoints (APIKeyHeader)                                     │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Rate limiting            │ slowapi on FastAPI to prevent abuse                                                      │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ PII in news              │ F1 news is public. No PII risk unless scraping social media comments with usernames      │
  └──────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘
  # Minimal API security
  from fastapi import FastAPI, Depends, HTTPException
  from fastapi.security import APIKeyHeader

  API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

  async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
      if api_key != os.getenv("RAG_API_KEY"):
          raise HTTPException(status_code=403, detail="Invalid API key")

  ---
  10. Prioritized TODO List

  Phase 1: Foundation (Week 1-2)

  - Install pgvector extension on PostgreSQL (CREATE EXTENSION vector) DONE
  - Create new RAG tables (news_articles, news_chunks, ingestion_log, retrieval_feedback) DONE
  - Install dependencies: sentence-transformers, trafilatura, feedparser, spacy, apscheduler DONE
  - Build llm/embeddings.py -- wrapper around all-MiniLM-L6-v2 
  - Build llm/chunker.py -- semantic chunking with F1-specific rules DONE
  - Centralize DB config from hardcoded strings to .env

  Phase 2: Ingestion Pipeline (Week 3-4)

  - Build RSS ingester for The Race, RaceFans, Motorsport.com, F1.com
  - Build driver code + circuit extraction (regex + lookup table)
  - Implement content dedup (URL + content hash + embedding similarity)
  - Build ingestion_log tracking
  - Manually ingest 100+ articles to seed the vector DB
  - Validate embedding quality (spot-check nearest neighbors make sense)

  Phase 3: Retrieval + LLM (Week 5-6)

  - Build llm/retriever.py -- hybrid search (cosine + metadata filters)
  - Build llm/context_builder.py -- merge ML predictions + news + weather
  - Build prompt templates for race analysis and fantasy team selection
  - Wire up Gemini Flash as primary LLM
  - Test end-to-end: query -> retrieve -> generate
  - Build 20+ labeled test queries for retrieval evaluation

  Phase 4: Automation + API (Week 7-8)

  - Set up APScheduler for RSS/Reddit ingestion
  - Build FastAPI REST endpoint: POST /api/ask
  - Add API key authentication
  - Build race-weekend-aware scheduling (more frequent ingestion Fri-Sun)
  - Add ingestion monitoring (alerts on failures)

  Phase 5: Evaluation + Optimization (Week 9-10)

  - Run retrieval evaluation (precision@5, MRR targets)
  - Implement re-ranking (cross-encoder or Cohere reranker)
  - Build retrieval feedback loop (user ratings)
  - Compare Gemini Flash vs Claude Haiku for answer quality
  - Tune chunk size and overlap based on retrieval metrics
  - Add Reddit and YouTube transcript ingestion

  Phase 6: Production Hardening (Week 11-12)

  - Add error handling and retry logic for all external APIs
  - Build dataset versioning (tag embedding model version with chunks)
  - Add monitoring dashboard (Grafana or simple HTML page)
  - Load test: simulate 100 concurrent queries
  - Document the full system for maintainability

  ---
  Recommended Tech Stack Summary
  Component: Vector DB
  Tool: pgvector (PostgreSQL extension)
  Reason: Already have PostgreSQL; native JOINs with your tables
  ────────────────────────────────────────
  Component: Embeddings
  Tool: all-MiniLM-L6-v2 (sentence-transformers)
  Reason: Free, local, 384-dim, fast on CPU
  ────────────────────────────────────────
  Component: Re-ranker
  Tool: cross-encoder/ms-marco-MiniLM-L-6-v2
  Reason: Free, boosts retrieval precision
  ────────────────────────────────────────
  Component: LLM (primary)
  Tool: Gemini 2.5 Flash
  Reason: Free tier, 1M context, fast
  ────────────────────────────────────────
  Component: LLM (deep analysis)
  Tool: Claude Sonnet 4.5
  Reason: Superior reasoning for complex queries
  ────────────────────────────────────────
  Component: RSS Parsing
  Tool: feedparser
  Reason: Battle-tested, handles edge cases
  ────────────────────────────────────────
  Component: HTML Cleaning
  Tool: trafilatura
  Reason: Best article text extraction
  ────────────────────────────────────────
  Component: NLP
  Tool: spaCy (en_core_web_sm)
  Reason: Sentence splitting, entity extraction
  ────────────────────────────────────────
  Component: Scheduler
  Tool: APScheduler
  Reason: Python-native, no external deps
  ────────────────────────────────────────
  Component: API
  Tool: FastAPI
  Reason: Async, auto-docs, easy to add
  ────────────────────────────────────────
  Component: Monitoring
  Tool: ingestion_log table + simple queries
  Reason: Keep it simple at your scale
  Your ML foundation (XGBoost + SHAP + risk flags) is designed to feed into RAG context already. The main work is building
  the ingestion and retrieval layers around it.                                            
                                         