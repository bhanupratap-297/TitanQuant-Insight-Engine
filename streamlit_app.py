# --- NEWS & SOCIAL TABS (paste inside your stock detail area) ---
import streamlit as st
from core.news import fetch_company_news, fetch_social_posts, process_headlines_for_ui

# finbert must already be loaded earlier in the script:
# e.g. finbert = load_finbert()

tab1, tab2 = st.tabs(["News headlines", "Social media (beta)"])

with tab1:
    # fetch headlines (provide your NEWS API key via st.secrets or pass None to get demo headlines)
    api_key = st.secrets.get("NEWSAPI_KEY") if "NEWSAPI_KEY" in st.secrets else None
    raw_articles = fetch_company_news(ticker, api_key=api_key, limit=12)
    # extract text list for the predictor
    headlines = [a.get("title") for a in raw_articles if a.get("title")]

    df_sent, sentiments, probs = process_headlines_for_ui(headlines, finbert, n_clusters=3)

    if df_sent.empty:
        st.info("No headlines available or News API not configured.")
    else:
        # show cluster selector and table
        st.subheader("Headline clusters (themes)")
        cluster_values = sorted(df_sent["Cluster"].unique())
        selected_cluster = st.selectbox("Select cluster to inspect", cluster_values)
        st.dataframe(df_sent[df_sent["Cluster"] == selected_cluster].reset_index(drop=True), height=300)

        # quick counts for labels
        counts = df_sent["Label"].value_counts().to_dict()
        st.write("**Sentiment counts:**", counts)

with tab2:
    posts = fetch_social_posts(ticker, platform="x", limit=50)
    if not posts:
        st.info("Social media integration not configured yet.")
    else:
        df_sent_social, sentiments_social, probs_social = process_headlines_for_ui(posts, finbert, n_clusters=3)
        st.subheader("Social media clusters")
        st.dataframe(df_sent_social, height=300)
