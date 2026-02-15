"""
llm/chunker.py -- Semantic chunking for F1 articles.

Takes a full article (e.g., 2000 words) and splits it into chunks of
~200-300 tokens each. Each chunk gets its own embedding and row in
the news_chunks table.

USAGE:
    from llm.chunker import chunk_article, extract_driver_codes, extract_circuit

    chunks = chunk_article("Full article text here...")
    # Returns: [{"text": "...", "token_count": 234}, {"text": "...", "token_count": 198}, ...]
"""

import spacy
import re
import trafilatura

# spaCy is a Natural Language Processing (NLP) library.
# We use it here for ONE job: splitting text into sentences.
# nlp() parses text into a Doc object. doc.sents gives sentence boundaries.
# It handles abbreviations ("Dr.", "Art. 3.5.1") and decimals ("1:29.4")
# correctly -- str.split(".") would butcher those.
nlp = spacy.load("en_core_web_sm")

MAX_CHUNK_TOKENS = 256
OVERLAP_SENTENCES = 2


def chunk_article(text: str, max_tokens: int = MAX_CHUNK_TOKENS, overlap: int = OVERLAP_SENTENCES) -> list[dict]:
    """
    Split an article into semantic chunks suitable for embedding.

    Args:
        text:       The full cleaned article text
        max_tokens: Maximum tokens per chunk (approximate, using word count)
        overlap:    Number of sentences to repeat at the start of each new chunk

    Returns:
        List of dicts: [{"text": "chunk text...", "token_count": 234}, ...]
    """
    # nlp(text) parses the text. doc.sents yields Span objects (one per sentence).
    # We filter out tiny fragments (< 10 chars) like "Fig. 1" or "Source: FIA".
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())

        # "Would adding this sentence blow past our token budget?"
        # AND "Do we actually have sentences to save?" (avoids saving empty chunks)
        if current_token_count + sentence_tokens > max_tokens and current_sentences:
            # Chunk is full -- save it
            chunks.append({
                "text": " ".join(current_sentences),
                "token_count": current_token_count,  # word count of this chunk
            })
            
            #preserve some sentences
            current_sentences = current_sentences[-overlap:] if overlap else []
            
            #compute token count based on the sentence overlaps
            current_token_count = sum(len(s.split()) for s in current_sentences)

        current_sentences.append(sentence)
        current_token_count += sentence_tokens

    if current_sentences:
        chunks.append({"text": " ".join(current_sentences), "token_count": current_token_count})

    return chunks


# --------------------------------------------------------------------------- #
# Metadata extraction helpers
# --------------------------------------------------------------------------- #

F1_DRIVER_CODES = {
    "verstappen": "VER", "norris": "NOR", "leclerc": "LEC",
    "sainz": "SAI", "hamilton": "HAM", "russell": "RUS",
    "piastri": "PIA", "alonso": "ALO", "stroll": "STR",
    "gasly": "GAS", "ocon": "OCO", "tsunoda": "TSU",
    "ricciardo": "RIC", "hulkenberg": "HUL", "magnussen": "MAG",
    "bottas": "BOT", "zhou": "ZHO", "albon": "ALB",
    "sargeant": "SAR", "perez": "PER", "bearman": "BEA",
    "lawson": "LAW", "colapinto": "COL", "doohan": "DOO",
    "antonelli": "ANT", "hadjar": "HAD", "bortoleto": "BOR",
}


def extract_driver_codes(text: str) -> list[str]:
    """
    Scan text for driver name mentions and return their 3-letter codes.

    Example:
        extract_driver_codes("Verstappen led from Hamilton and Norris")
        -> ["VER", "HAM", "NOR"]
    """
    codes = []
    text_lower = text.lower()
    for name in F1_DRIVER_CODES:
        if name in text_lower:
            codes.append(F1_DRIVER_CODES[name])
    return codes


# Maps keyword mentions to canonical circuit names (should match your races table).
F1_CIRCUITS = {
    "bahrain": "Bahrain International Circuit",
    "sakhir": "Bahrain International Circuit",
    "jeddah": "Jeddah Corniche Circuit",
    "saudi": "Jeddah Corniche Circuit",
    "albert park": "Albert Park Grand Prix Circuit",
    "melbourne": "Albert Park Grand Prix Circuit",
    "suzuka": "Suzuka Circuit",
    "shanghai": "Shanghai International Circuit",
    "miami": "Miami International Autodrome",
    "imola": "Autodromo Enzo e Dino Ferrari",
    "monaco": "Circuit de Monaco",
    "montreal": "Circuit Gilles Villeneuve",
    "canada": "Circuit Gilles Villeneuve",
    "barcelona": "Circuit de Barcelona-Catalunya",
    "red bull ring": "Red Bull Ring",
    "spielberg": "Red Bull Ring",
    "austria": "Red Bull Ring",
    "silverstone": "Silverstone Circuit",
    "hungaroring": "Hungaroring",
    "budapest": "Hungaroring",
    "spa": "Circuit de Spa-Francorchamps",
    "zandvoort": "Circuit Zandvoort",
    "monza": "Autodromo Nazionale di Monza",
    "baku": "Baku City Circuit",
    "marina bay": "Marina Bay Street Circuit",
    "singapore": "Marina Bay Street Circuit",
    "cota": "Circuit of the Americas",
    "austin": "Circuit of the Americas",
    "mexico city": "Autodromo Hermanos Rodriguez",
    "interlagos": "Autodromo Jose Carlos Pace",
    "sao paulo": "Autodromo Jose Carlos Pace",
    "las vegas": "Las Vegas Strip Street Circuit",
    "lusail": "Lusail International Circuit",
    "qatar": "Lusail International Circuit",
    "yas marina": "Yas Marina Circuit",
    "abu dhabi": "Yas Marina Circuit",
}


def extract_circuit(text: str) -> str | None:
    """Identify which circuit an article is about. Returns canonical name or None."""
    text_lower = text.lower()
    for keyword, circuit_name in F1_CIRCUITS.items():
        if keyword in text_lower:
            return circuit_name
    return None


def clean_article_text(raw_html: str) -> str:
    """Extract clean article text from raw HTML using trafilatura."""
    # trafilatura.extract() returns the main article text from HTML,
    # stripping nav bars, ads, footers, cookie banners, etc.
    # You must capture the return value -- it returns a NEW string.
    cleaned = trafilatura.extract(raw_html)
    if cleaned:
        return cleaned

    # Fallback if trafilatura returns None (e.g., plain text input, not HTML)
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = re.sub(r"\s+", " ", text).strip()
    return text