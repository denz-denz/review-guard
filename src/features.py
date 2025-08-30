from sentence_transformers import SentenceTransformer, util
from .utils import PlaceProfile
from typing import Dict

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def relevancy(self, review_text: str, place: PlaceProfile) -> float:
        if not review_text:
            return 0.0
        a = self.model.encode(review_text, normalize_embeddings=True)
        b = self.model.encode(place.as_text(), normalize_embeddings=True)
        sim = float(util.cos_sim(a, b))
        # map [-1,1] to [0,1]
        return (sim + 1.0) / 2.0
