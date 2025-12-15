
"""
Data Fetcher Module for Stock Market Data
data_fetcher.py
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import feedparser

class DataFetcher:
    def __init__(self):
        """Initialize data fetcher"""
        self.cache = {}
    
    def get_stock_data(self, symbol, period='1y', interval='1d'):
        """
        Fetch stock price data using yfinance
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"No data found for {symbol}")
                return None
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_stock_info(self, symbol):
        """
        Get detailed stock information
        
        Returns:
            Dictionary with stock info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return None
    
    def get_stock_news(self, symbol, count=20):
        """
        Fetch news articles for a stock from multiple sources
        
        Returns:
            List of news articles with title, summary, link, published date
        """
        news_articles = []
        
        # Try Yahoo Finance news
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for article in news[:count]:
                news_articles.append({
                    'title': article.get('title', ''),
                    'summary': article.get('summary', article.get('title', '')),
                    'link': article.get('link', ''),
                    'published': datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%Y-%m-%d'),
                    'source': 'Yahoo Finance'
                })
        except Exception as e:
            print(f"Error fetching Yahoo Finance news: {e}")
        
        # Try Google News RSS
        try:
            company_name = self._get_company_name(symbol)
            rss_url = f"https://news.google.com/rss/search?q={company_name}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:count]:
                news_articles.append({
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', entry.get('title', '')),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': 'Google News'
                })
        except Exception as e:
            print(f"Error fetching Google News: {e}")
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        for article in news_articles:
            if article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                unique_articles.append(article)
        
        return unique_articles[:count]
    
    def _get_company_name(self, symbol):
        """Get company name from symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('longName', symbol)
        except:
            return symbol
    
    def get_financial_statements(self, symbol):
        """
        Get financial statements (income statement, balance sheet, cash flow)
        """
        try:
            ticker = yf.Ticker(symbol)
            
            return {
                'income_statement': ticker.income_stmt,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cash_flow,
                'quarterly_income': ticker.quarterly_income_stmt,
                'quarterly_balance': ticker.quarterly_balance_sheet,
                'quarterly_cash_flow': ticker.quarterly_cash_flow
            }
        except Exception as e:
            print(f"Error fetching financial statements: {e}")
            return None
    
    def get_market_indices(self):
        """
        Get data for major market indices
        """
        indices = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "Dow Jones": "^DJI",
            "Nifty 50": "^NSEI",
            "Nikkei 225": "^N225",
            "FTSE 100": "^FTSE",
            "DAX": "^GDAXI",
            "Hang Seng": "^HSI"
        }
        
        market_data = {}
        for name, symbol in indices.items():
            data = self.get_stock_data(symbol, period='1d')
            if data is not None and len(data) > 0:
                market_data[name] = {
                    'symbol': symbol,
                    'current': data['Close'].iloc[-1],
                    'change': data['Close'].iloc[-1] - data['Open'].iloc[0],
                    'change_pct': ((data['Close'].iloc[-1] - data['Open'].iloc[0]) / data['Open'].iloc[0]) * 100
                }
        
        return market_data
    
    def get_realtime_quote(self, symbol):
        """
        Get real-time quote data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'change': info.get('regularMarketChange', 0),
                'change_pct': info.get('regularMarketChangePercent', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            print(f"Error fetching real-time quote: {e}")
            return None
