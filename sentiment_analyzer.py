"""
Advanced Sentiment Analysis using FinBERT
sentiment_analyzer.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        """Initialize FinBERT model for financial sentiment analysis"""
        self.model_name = "ProsusAI/finbert"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            self.tokenizer = None
            self.model = None
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a single text using FinBERT
        Returns: sentiment score (-1 to 1) and label
        """
        if not self.model or not self.tokenizer:
            return {'score': 0.0, 'label': 'NEUTRAL'}
        
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                                  truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [positive, negative, neutral]
            positive = predictions[0][0].item()
            negative = predictions[0][1].item()
            neutral = predictions[0][2].item()
            
            # Calculate compound score
            score = positive - negative
            
            # Determine label
            if score > 0.1:
                label = 'POSITIVE'
            elif score < -0.1:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
            
            return {
                'score': score,
                'label': label,
                'positive': positive,
                'negative': negative,
                'neutral': neutral
            }
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {'score': 0.0, 'label': 'NEUTRAL'}
    
    def analyze_news_batch(self, news_list):
        """
        Analyze sentiment for a batch of news articles
        """
        sentiments = []
        articles_data = []
        
        for article in news_list:
            title = article.get('title', '')
            summary = article.get('summary', '')
            
            # Combine title and summary
            text = f"{title}. {summary}"
            
            # Analyze sentiment
            sentiment = self.analyze_text(text)
            sentiments.append(sentiment['score'])
            
            articles_data.append({
                'title': title,
                'summary': summary,
                'sentiment': sentiment['score'],
                'label': sentiment['label'],
                'published': article.get('published', ''),
                'link': article.get('link', '')
            })
        
        # Calculate overall metrics
        sentiments_array = np.array(sentiments)
        overall_sentiment = np.mean(sentiments_array)
        
        # Count sentiments
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Determine overall label
        if overall_sentiment > 0.1:
            sentiment_label = "🟢 BULLISH"
        elif overall_sentiment < -0.1:
            sentiment_label = "🔴 BEARISH"
        else:
            sentiment_label = "🟡 NEUTRAL"
        
        # Sentiment timeline (group by date)
        timeline = self._create_sentiment_timeline(articles_data)
        
        # Clustering analysis
        clusters = self._perform_clustering(articles_data)
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_label': sentiment_label,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'articles': articles_data,
            'sentiment_timeline': timeline,
            'clusters': clusters
        }
    
    def _create_sentiment_timeline(self, articles_data):
        """Create sentiment timeline grouped by date"""
        try:
            df = pd.DataFrame(articles_data)
            if 'published' not in df.columns or df['published'].isna().all():
                return []
            
            # Convert published to datetime
            df['date'] = pd.to_datetime(df['published'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            # Group by date and calculate average sentiment
            timeline = df.groupby(df['date'].dt.date)['sentiment'].mean().reset_index()
            timeline.columns = ['date', 'sentiment']
            timeline = timeline.sort_values('date')
            
            return timeline.to_dict('records')
        except Exception as e:
            print(f"Error creating timeline: {e}")
            return []
    
    def _perform_clustering(self, articles_data, n_clusters=3):
        """Perform clustering on articles based on content similarity"""
        try:
            if len(articles_data) < n_clusters:
                return {}
            
            # Extract texts
            texts = [f"{a['title']} {a['summary']}" for a in articles_data]
            
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X = vectorizer.fit_transform(texts)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # Analyze each cluster
            cluster_analysis = {}
            for i in range(n_clusters):
                cluster_articles = [articles_data[j] for j in range(len(articles_data)) if clusters[j] == i]
                
                if cluster_articles:
                    cluster_sentiments = [a['sentiment'] for a in cluster_articles]
                    avg_sentiment = np.mean(cluster_sentiments)
                    
                    # Extract theme (most common words)
                    cluster_texts = [f"{a['title']}" for a in cluster_articles]
                    cluster_vectorizer = TfidfVectorizer(max_features=3, stop_words='english')
                    cluster_tfidf = cluster_vectorizer.fit_transform(cluster_texts)
                    theme_words = cluster_vectorizer.get_feature_names_out()
                    theme = " | ".join(theme_words)
                    
                    cluster_analysis[i] = {
                        'theme': theme,
                        'count': len(cluster_articles),
                        'avg_sentiment': avg_sentiment,
                        'sample_headlines': [a['title'] for a in cluster_articles[:5]]
                    }
            
            return cluster_analysis
        except Exception as e:
            print(f"Error in clustering: {e}")
            return {}
