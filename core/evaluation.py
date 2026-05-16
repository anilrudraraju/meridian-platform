import streamlit as st

BASE_MODEL       = "gpt-3.5-turbo-0125"
FINE_TUNED_MODEL = "ft:gpt-3.5-turbo-0125:personal::DZTJSppd"


@st.cache_resource
def load_evaluator():
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2"), rouge_scorer.RougeScorer(["rouge1", "rougeL"])


class FinancialEvaluator:
    """Source: week4_capstone.ipynb"""

    def __init__(self):
        self.embedding_model, self.rouge = load_evaluator()

    def evaluate_semantic_similarity(self, pred: str, ref: str) -> float:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        pred_emb = self.embedding_model.encode([pred])
        ref_emb  = self.embedding_model.encode([ref])
        return float(cosine_similarity(pred_emb, ref_emb)[0][0])

    def check_compliance(self, text: str) -> float:
        required = ["past performance", "does not guarantee"]
        found = [p for p in required if p in text.lower()]
        return len(found) / len(required)
