# core/recommendation.py
import numpy as np

def combine_scores(
    sentiment_score: float,  # 0–1 (FinBERT-based)
    rsi_value: float,        # 0–100
    trend: str,              # "Uptrend", "Downtrend", "Sideways"
) -> dict:
    """
    Very simple heuristic scoring (for a student project).
    """

    # Normalize RSI: middle (40–60) is neutral.
    if rsi_value < 30:
        rsi_score = 0.2  # oversold: good to buy
    elif rsi_value > 70:
        rsi_score = 0.2  # overbought: risky
    else:
        rsi_score = 0.6  # neutral to ok

    trend_score = {
        "Uptrend": 0.8,
        "Sideways": 0.5,
        "Downtrend": 0.2,
    }.get(trend, 0.5)

    # Weighted combination
    total = 0.5 * sentiment_score + 0.3 * trend_score + 0.2 * rsi_score

    # Convert to label
    if total >= 0.75:
        label = "Strong Buy"
    elif total >= 0.6:
        label = "Buy"
    elif total >= 0.4:
        label = "Hold"
    elif total >= 0.25:
        label = "Sell"
    else:
        label = "Strong Sell"

    return {
        "score": float(total),
        "label": label,
        "components": {
            "sentiment": float(sentiment_score),
            "trend": float(trend_score),
            "rsi": float(rsi_score),
        },
    }

from core.recommendation import combine_scores
from ui.components import sentiment_gauge

# after computing overall_sentiment_score, stock_data with RSI & Trend
rsi_value = float(stock_data["RSI14"].iloc[-1])
trend = stock_data["Trend"].iloc[-1]

rec = combine_scores(overall_sentiment_score, rsi_value, trend)

st.subheader("📊 Overall Recommendation")
sentiment_gauge(rec["score"], key="overall_rec")
st.markdown(f"**Signal:** {rec['label']}")
st.json(rec["components"])
