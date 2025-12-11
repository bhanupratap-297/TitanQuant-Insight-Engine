# ui/components.py
import plotly.graph_objects as go
import streamlit as st

def sentiment_gauge(score: float, key: str | None = None):
    """
    score in [0, 1]; map to 0–100 for gauge
    """
    value = score * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%"},
        title={"text": "Bullishness"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.3},
            "steps": [
                {"range": [0, 30], "color": "#ff4b4b"},     # strong sell
                {"range": [30, 50], "color": "#f9c74f"},    # cautious / hold
                {"range": [50, 70], "color": "#90be6d"},    # buy
                {"range": [70, 100], "color": "#43aa8b"},   # strong buy
            ],
        },
    ))
    st.plotly_chart(fig, use_container_width=True, key=key)

#-------------------------------

# inside app when you have FinBERT results
sentiments, probs = finbert.predict(headlines)
# Take weighted average: 1*Positive + 0.5*Neutral + 0*Negative
weights = np.array([0.0, 0.5, 1.0])
headline_scores = probs @ weights
overall_score = float(headline_scores.mean())
sentiment_gauge(overall_score, key="headline_gauge")
