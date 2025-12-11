# core/sentiment.py
import numpy as np
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.cluster import KMeans

FINBERT_MODEL_NAME = "ProsusAI/finbert"  # or "yiyanghkust/finbert-tone"

class FinbertSentiment:
    def __init__(self, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Prosus FinBERT label order is usually [negative, neutral, positive]
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    @torch.inference_mode()
    def predict(self, texts: list[str]):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        sentiments = []
        for t, p in zip(texts, probs):
            label_id = int(np.argmax(p))
            sentiments.append({
                "text": t,
                "label": self.id2label[label_id].title(),
                "scores": {
                    "Negative": float(p[0]),
                    "Neutral": float(p[1]),
                    "Positive": float(p[2]),
                }
            })
        return sentiments, probs

def basic_textblob_score(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def cluster_headlines(headlines: list[str], probs: np.ndarray, n_clusters: int = 3):
    """
    Simple clustering of headlines based on FinBERT probabilities
    -> returns cluster labels; later you can interpret clusters manually.
    """
    if len(headlines) < n_clusters:
        n_clusters = len(headlines)

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(probs)  # cluster in sentiment-probability space
    return labels
