"""External search resources; models load only when first used."""
import json
import os
import numpy as np
from dotenv import load_dotenv

MODEL_ID = "jinaai/jina-embeddings-v2-base-en"
_model = None
_nlp = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    return _model

def nlp(text):
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp(text)

def embed(text: str):
    try:
        model = get_model()
        input_text = text[:8000]
        vec = model.encode(input_text)
        return np.array(vec)
    except Exception as e:
        print(f"❌ Local Inference failed: {e}")
        return np.zeros(768)

load_dotenv()
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
db = None
collection = None
if ASTRA_TOKEN and ASTRA_ENDPOINT:
    from astrapy import DataAPIClient
    client = DataAPIClient(ASTRA_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
    collection = db.get_collection("movies")

PEOPLE_INDEX_PATH = os.path.join(os.path.dirname(__file__), "people_index.json")
try:
    with open(PEOPLE_INDEX_PATH, encoding="utf-8") as f:
        PEOPLE_SET = set(json.load(f))
except (OSError, ValueError):
    PEOPLE_SET = set()


_hnsw_index = None

def get_hnsw_index():
    """Lazily initialize in-memory HNSW index for local ANN queries."""
    global _hnsw_index
    if _hnsw_index is None:
        from engine.hnsw import HNSWIndex
        _hnsw_index = HNSWIndex(dim=768, m=16, ef_construction=64, ef_search=32)
    return _hnsw_index
