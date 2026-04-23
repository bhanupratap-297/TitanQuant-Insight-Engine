"""
Multi-Source News Aggregator
multi_news_aggregator.py
Fetches news from: Yahoo Finance, Google News, NewsAPI, Finnhub, Alpha Vantage, Seeking Alpha
"""

import yfinance as yf
import feedparser
import requests
import pandas as pd
from datetime import datetime, timedelta
import json

class MultiNewsAggregator:
    def __init__(self):
        """Initialize news aggregator with API keys"""
        self.newsapi_key = "your_newsapi_key"  # Get from newsapi.org
        self.finnhub_key = "your_finnhub_key"  # Get from finnhub.io
        self.alpha_vantage_key = "your_alpha_vantage_key"  # Get from alphavantage.co
    
    def fetch_all_news(self, symbol, count=50):
        """
        Fetch news from all available sources
        Returns combined list of unique news articles
        """
        all_news = []
        
        # Yahoo Finance News
        yahoo_news = self._fetch_yahoo_news(symbol, count)
        all_news.extend(yahoo_news)
        
        # Google News RSS
        google_news = self._fetch_google_news(symbol, count)
        all_news.extend(google_news)
        
        # NewsAPI.org
        newsapi_news = self._fetch_newsapi_news(symbol, count)
        all_news.extend(newsapi_news)
        
        # Finnhub API
        finnhub_news = self._fetch_finnhub_news(symbol, count)
        all_news.extend(finnhub_news)
        
        # Seeking Alpha (RSS)
        seeking_alpha_news = self._fetch_seeking_alpha_news(symbol, count)
        all_news.extend(seeking_alpha_news)
        
        # Remove duplicates based on title + source
        unique_news = {}
        for article in all_news:
            key = f"{article['title']}_{article['source']}"
            if key not in unique_news:
                unique_news[key] = article
        
        # Sort by date (newest first)
        sorted_news = sorted(unique_news.values(), 
                            key=lambda x: datetime.fromisoformat(x['published'].replace('Z', '+00:00')) 
                            if x['published'] else datetime.now(), 
                            reverse=True)
        
        return sorted_news[:count]
    
    def _fetch_yahoo_news(self, symbol, count=20):
        """Fetch news from Yahoo Finance"""
        articles = []
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for article in news[:count]:
                articles.append({
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'link': article.get('link', ''),
                    'published': datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat(),
                    'source': 'Yahoo Finance',
                    'image': article.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', ''),
                    'publisher': article.get('publisher', '')
                })
        except Exception as e:
            print(f"Error fetching Yahoo Finance news: {e}")
        
        return articles
    
    def _fetch_google_news(self, symbol, count=20):
        """Fetch news from Google News RSS"""
        articles = []
        try:
            company_name = self._get_company_name(symbol)
            rss_url = f"https://news.google.com/rss/search?q={company_name}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:count]:
                articles.append({
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', datetime.now().isoformat()),
                    'source': 'Google News',
                    'image': '',
                    'publisher': 'Google News'
                })
        except Exception as e:
            print(f"Error fetching Google News: {e}")
        
        return articles
    
    def _fetch_newsapi_news(self, symbol, count=20):
        """Fetch news from NewsAPI.org"""
        articles = []
        try:
            if self.newsapi_key == "your_newsapi_key":
                return articles
            
            url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&language=en&pageSize={count}"
            headers = {'X-Api-Key': self.newsapi_key}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for article in data.get('articles', [])[:count]:
                    articles.append({
                        'title': article.get('title', ''),
                        'summary': article.get('description', ''),
                        'link': article.get('url', ''),
                        'published': article.get('publishedAt', datetime.now().isoformat()),
                        'source': 'NewsAPI',
                        'image': article.get('urlToImage', ''),
                        'publisher': article.get('source', {}).get('name', '')
                    })
        except Exception as e:
            print(f"Error fetching NewsAPI news: {e}")
        
        return articles
    
    def _fetch_finnhub_news(self, symbol, count=20):
        """Fetch news from Finnhub.io"""
        articles = []
        try:
            if self.finnhub_key == "your_finnhub_key":
                return articles
            
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={self._get_date_range()}&to={datetime.now().strftime('%Y-%m-%d')}&token={self.finnhub_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for article in data[:count]:
                    articles.append({
                        'title': article.get('headline', ''),
                        'summary': article.get('summary', ''),
                        'link': article.get('url', ''),
                        'published': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                        'source': 'Finnhub',
                        'image': article.get('image', ''),
                        'publisher': article.get('source', '')
                    })
        except Exception as e:
            print(f"Error fetching Finnhub news: {e}")
        
        return articles
    
    def _fetch_seeking_alpha_news(self, symbol, count=20):
        """Fetch news from Seeking Alpha"""
        articles = []
        try:
            rss_url = "https://feeds.seekingalpha.com/feed.xml"
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:count]:
                if symbol.lower() in entry.get('title', '').lower() or symbol.lower() in entry.get('summary', '').lower():
                    articles.append({
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', datetime.now().isoformat()),
                        'source': 'Seeking Alpha',
                        'image': '',
                        'publisher': 'Seeking Alpha'
                    })
        except Exception as e:
            print(f"Error fetching Seeking Alpha news: {e}")
        
        return articles
    
    def categorize_news_by_theme(self, news_list):
        """Categorize news articles by theme"""
        categories = {
            'earnings': [],
            'merger_acquisition': [],
            'product': [],
            'regulatory': [],
            'market': [],
            'other': []
        }
        
        keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'q1', 'q2', 'q3', 'q4', 'quarterly'],
            'merger_acquisition': ['merger', 'acquisition', 'deal', 'buyout', 'acquired'],
            'product': ['launch', 'product', 'announcement', 'release'],
            'regulatory': ['sec', 'regulatory', 'compliance', 'fda', 'lawsuit'],
            'market': ['stock', 'price', 'market', 'rally', 'decline', 'surge']
        }
        
        for article in news_list:
            title_lower = article['title'].lower() + ' ' + article['summary'].lower()
            categorized = False
            
            for category, keywords_list in keywords.items():
                if any(keyword in title_lower for keyword in keywords_list):
                    categories[category].append(article)
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(article)
        
        return categories
    
    def get_trending_stocks_news(self):
        """Get trending stocks and their news""" 
        trending = {}
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        
        for symbol in symbols:
            news = self.fetch_all_news(symbol, count=5)
            if news:
                trending[symbol] = news
        
        return trending
    
    def export_to_csv(self, news_list, filename='news_export.csv'):
        """Export news to CSV file""" 
        df = pd.DataFrame(news_list)
        df.to_csv(filename, index=False)
        return filename
    
    def export_to_json(self, news_list, filename='news_export.json'):
        """Export news to JSON file""" 
        with open(filename, 'w') as f:
            json.dump(news_list, f, indent=2)
        return filename
    
    def _get_company_name(self, symbol):
        """Get company name from symbol""" 
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('longName', symbol)
        except:
            return symbol
    
    def _get_date_range(self, days=30):
        """Get date range for API requests""" 
        date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return date
    
    def get_news_summary(self, news_list):
        """Get summary statistics of news articles""" 
        df = pd.DataFrame(news_list)
        
        return {
            'total_articles': len(news_list),
            'sources': df['source'].value_counts().to_dict(),
            'date_range': {
                'earliest': df['published'].min(),
                'latest': df['published'].max()
            },
            'publishers': df['publisher'].value_counts().head(5).to_dict()
        }