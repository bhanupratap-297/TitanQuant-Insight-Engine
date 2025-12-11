# core/news.py
from typing import Literal, List, Tuple, Dict, Any
import requests
import pandas as pd
from core.sentiment import cluster_headlines

# Note: This module should NOT import streamlit or contain UI code.
# It only fetches/processes data and returns Python objects for the UI to render.

def fetch_company_news(company: str, api_key: str = None, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch latest headlines for `company`. This is a minimal NewsAPI-style stub.
    Replace with your actual NewsAPI / RSS implementation.
    Returns list of dicts: [{"title": "...", "source": "...", "publishedAt": "...", "url": "..."}]
    """
    # If no API key provided, return empty list (or sample headlines)
    if not api_key:
        # sample fallback headlines for demo
        sample = [
            {"title": f"{company}: Company announces new product line", "source": "Demo", "publishedAt": None, "url": None},
            {"title": f"{company}: Quarterly earnings beat expectations", "source": "Demo", "publishedAt": None, "url": None},
            {"title": f"{company}: CEO interview hints at expansion", "source": "Demo", "publishedAt": None, "url": None},
        ]
        return sample[:limit]

    # Example NewsAPI.org request (uncomment + adapt if you will use it)
    # url = "https://newsapi.org/v2/everything"
    # params = {
    #     "q": company,
    #     "pageSize": limit,
    #     "sortBy": "relevancy",
    #     "language": "en",
    #     "apiKey": api_key,
    # }
    # resp = requests.get(url, params=params, timeout=10)
    # resp.raise_for_status()
    # data = resp.json()
    # articles = data.get("articles", [])
    # simplified = []
    # for a in articles:
    #     simplified.append({
    #         "title": a.get("title"),
    #         "source": a.get("source", {}).get("name"),
    #         "publishedAt": a.get("publishedAt"),
    #         "url": a.get("url")
    #     })
    # return simplified

    # If you reach here but haven't implemented API, return empty list
    return []


def fetch_social_posts(company: str, platform: Literal["x", "reddit"] = "x", limit: int = 50) -> List[str]:
    """
    Stub for social posts. Hook your Tweepy / X API or PRAW (Reddit) here.
    Return a list of post texts (strings).
    """
    # Placeholder: empty list meaning "not configured"
    return []


def process_headlines_for_ui(
    headlines: List[str],
    finbert_predictor,   # instance of FinbertSentiment (or any object with .predict(texts) -> (sentiments, probs))
    n_clusters: int = 3
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Any]:
    """
    Given a list of headline strings and a FinBERT-like predictor object,
    return:
      - df_sent: DataFrame ready for UI display (columns: Headline, Label, Pos, Neu, Neg, Cluster)
      - sentiments: raw sentiments list returned by predictor
      - probs: the probability vectors returned by predictor (numpy array)
    """
    if not headlines:
        return pd.DataFrame(columns=["Headline", "Label", "Pos", "Neu", "Neg", "Cluster"]), [], None

    # call the predictor: expected to return (sentiments, probs)
    sentiments, probs = finbert_predictor.predict(headlines)

    # cluster headlines using probability vectors
    if probs is None:
        labels = [0] * len(headlines)
    else:
        try:
            labels = cluster_headlines(headlines, probs, n_clusters=n_clusters)
        except Exception:
            labels = [0] * len(headlines)

    rows = []
    for s, c in zip(sentiments, labels):
        rows.append({
            "Headline": s.get("text"),
            "Label": s.get("label"),
            "Pos": s.get("scores", {}).get("Positive", None),
            "Neu": s.get("scores", {}).get("Neutral", None),
            "Neg": s.get("scores", {}).get("Negative", None),
            "Cluster": f"Theme {int(c)}"
        })

    df_sent = pd.DataFrame(rows)
    return df_sent, sentiments, probs
