#Advanced Stock Market Sentiment Analyzer
#Main Application File - app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sentiment_analyzer import SentimentAnalyzer
from data_fetcher import DataFetcher
from ml_predictor import MLPredictor
from technical_analyzer import TechnicalAnalyzer
from visualizations import create_gauge_chart, create_candlestick_chart
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Advanced Stock Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS added here...
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(to bottom right, #1a1a2e, #16213e);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def initialize_components():
    return {
        'data_fetcher': DataFetcher(),
        'sentiment_analyzer': SentimentAnalyzer(),
        'ml_predictor': MLPredictor(),
        'technical_analyzer': TechnicalAnalyzer()
    }

components = initialize_components()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/stocks.png", width=150)
    st.title("🎯 Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Market Overview", "🔍 Stock Analysis", "📊 Technical Analysis", 
         "🤖 ML Predictions", "💭 Sentiment Dashboard"]
    )
    
    st.markdown("---")
    st.info("💡 **Tip**: Use the search bar to analyze any stock in detail!")

# Page: Market Overview
if page == "🏠 Market Overview":
    st.title("🌍 Global Market Overview")
    
    # Market selector
    markets = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "Nifty 50": "^NSEI",
        "Nikkei 225": "^N225",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "Hang Seng": "^HSI"
    }
    
    #Major indices in World
    st.subheader("📈 Major Indices")
    cols = st.columns(4)
    
    for idx, (name, symbol) in enumerate(list(markets.items())[:4]):
        with cols[idx % 4]:
            try:
                data = components['data_fetcher'].get_stock_data(symbol, period='1d')
                if data is not None and len(data) > 0:
                    current = data['Close'].iloc[-1]
                    prev = data['Open'].iloc[0]
                    change = ((current - prev) / prev) * 100
                    
                    st.metric(
                        label=name,
                        value=f"${current:,.2f}",
                        delta=f"{change:.2f}%"
                    )
            except:
                st.metric(label=name, value="N/A", delta="0%")
    
    st.markdown("---")
    
    # Market selection for detailed view
    selected_market = st.selectbox("🎯 Select Market for Detailed View", list(markets.keys()))
    market_symbol = markets[selected_market]
    
    # Get top companies for selected market
    st.subheader(f"🏆 Top Companies in {selected_market}")
    
    # Market-specific company lists
    market_companies = {
        "S&P 500": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "JNJ"],
        "NASDAQ": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST", "PEP"],
        "Dow Jones": ["AAPL", "MSFT", "UNH", "GS", "HD", "CAT", "MCD", "V", "AMGN", "BA"],
        "Nifty 50": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", 
                     "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LT.NS"],
        "Nikkei 225": ["7203.T", "6758.T", "9984.T", "6861.T", "6902.T", "8306.T", "9433.T", "4502.T", "6501.T", "6367.T"],
        "FTSE 100": ["SHEL.L", "AZN.L", "HSBA.L", "BP.L", "GSK.L", "DGE.L", "RIO.L", "ULVR.L", "NG.L", "LSEG.L"],
        "DAX": ["SAP.DE", "SIE.DE", "ALV.DE", "AIR.DE", "BAS.DE", "VOW3.DE", "DTE.DE", "MBG.DE", "BMW.DE", "MUV2.DE"],
        "Hang Seng": ["0700.HK", "9988.HK", "1299.HK", "0941.HK", "3690.HK", "2318.HK", "0388.HK", "1398.HK", "0005.HK", "0883.HK"]
    }
    
    companies = market_companies.get(selected_market, market_companies["S&P 500"])
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Market Performance", "🎯 Top Gainers & Losers", "🔝 By Market Cap"])
    
    with tab1:
        # Market performance chart
        market_data = components['data_fetcher'].get_stock_data(market_symbol, period='1mo')
        if market_data is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=market_data.index,
                y=market_data['Close'],
                mode='lines',
                name=selected_market,
                line=dict(color='#667eea', width=3),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.3)'
            ))
            fig.update_layout(
                title=f"{selected_market} - 30 Day Performance",
                template='plotly_dark',
                height=400,
                xaxis_title="Date",
                yaxis_title="Price"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Top Gainers & Losers")
        
        # Fetch data for all companies
        performance_data = []
        for symbol in companies:
            try:
                data = components['data_fetcher'].get_stock_data(symbol, period='1d')
                if data is not None and len(data) > 1:
                    current = data['Close'].iloc[-1]
                    prev = data['Close'].iloc[-2]
                    change = ((current - prev) / prev) * 100
                    performance_data.append({
                        'Symbol': symbol,
                        'Price': current,
                        'Change %': change
                    })
            except:
                continue
        
        if performance_data:
            df_performance = pd.DataFrame(performance_data)
            df_performance = df_performance.sort_values('Change %', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🟢 Top Gainers")
                top_gainers = df_performance.head(5)
                for _, row in top_gainers.iterrows():
                    st.success(f"**{row['Symbol']}**: ${row['Price']:.2f} ({row['Change %']:+.2f}%)")
            
            with col2:
                st.markdown("### 🔴 Top Losers")
                top_losers = df_performance.tail(5)
                for _, row in top_losers.iterrows():
                    st.error(f"**{row['Symbol']}**: ${row['Price']:.2f} ({row['Change %']:+.2f}%)")
    
    with tab3:
        st.subheader("💰 Companies by Market Cap")
        
        market_cap_data = []
        for symbol in companies[:10]:
            try:
                info = components['data_fetcher'].get_stock_info(symbol)
                if info:
                    market_cap_data.append({
                        'Symbol': symbol,
                        'Company': info.get('longName', symbol),
                        'Market Cap': info.get('marketCap', 0),
                        'Price': info.get('currentPrice', 0)
                    })
            except:
                continue
        
        if market_cap_data:
            df_market_cap = pd.DataFrame(market_cap_data)
            df_market_cap = df_market_cap.sort_values('Market Cap', ascending=False)
            df_market_cap['Market Cap (B)'] = (df_market_cap['Market Cap'] / 1e9).round(2)
            
            # Display as table
            st.dataframe(
                df_market_cap[['Symbol', 'Company', 'Market Cap (B)', 'Price']],
                use_container_width=True,
                hide_index=True
            )

# Page: Stock Analysis
elif page == "🔍 Stock Analysis":
    st.title("🔍 Detailed Stock Analysis")
    
    # Stock search
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_symbol = st.text_input("🔎 Enter Stock Symbol", "AAPL", help="Enter stock ticker (e.g., AAPL, GOOGL, RELIANCE.NS)")
    with col2:
        period = st.selectbox("📅 Period", ["1mo", "3mo", "6mo", "1y", "2y"])
    
    if stock_symbol:
        # Fetch stock data
        stock_data = components['data_fetcher'].get_stock_data(stock_symbol, period=period)
        stock_info = components['data_fetcher'].get_stock_info(stock_symbol)
        
        if stock_data is not None and stock_info:
            # Display key metrics
            st.subheader(f"📊 {stock_info.get('longName', stock_symbol)}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            current_price = stock_data['Close'].iloc[-1]
            prev_price = stock_data['Close'].iloc[-2]
            change_pct = ((current_price - prev_price) / prev_price) * 100
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
            with col2:
                st.metric("Market Cap", f"${stock_info.get('marketCap', 0)/1e9:.2f}B")
            with col3:
                st.metric("P/E Ratio", f"{stock_info.get('trailingPE', 0):.2f}")
            with col4:
                st.metric("52W High", f"${stock_info.get('fiftyTwoWeekHigh', 0):.2f}")
            with col5:
                st.metric("52W Low", f"${stock_info.get('fiftyTwoWeekLow', 0):.2f}")
            
            st.markdown("---")
            
            # Candlestick chart
            st.subheader("📈 Price Chart")
            fig = create_candlestick_chart(stock_data, stock_symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            fig_volume = go.Figure()
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(stock_data['Close'], stock_data['Open'])]
            fig_volume.add_trace(go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                marker_color=colors,
                name='Volume'
            ))
            fig_volume.update_layout(
                title="Trading Volume",
                template='plotly_dark',
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        else:
            st.error("❌ Unable to fetch stock data. Please check the symbol and try again.")

# Page: Technical Analysis
elif page == "📊 Technical Analysis":
    st.title("📊 Technical Analysis Dashboard")
    
    stock_symbol = st.text_input("🔎 Enter Stock Symbol", "AAPL")
    
    if stock_symbol:
        stock_data = components['data_fetcher'].get_stock_data(stock_symbol, period='1y')
        
        if stock_data is not None:
            # Calculate technical indicators
            tech_data = components['technical_analyzer'].calculate_all_indicators(stock_data)
            signals = components['technical_analyzer'].generate_signals(tech_data)
            
            # Display overall signal
            st.subheader("🎯 Trading Signal")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                overall_signal = signals['overall_signal']
                signal_color = "🟢" if overall_signal == "BUY" else "🔴" if overall_signal == "SELL" else "🟡"
                st.markdown(f"### {signal_color} {overall_signal}")
            
            with col2:
                st.metric("Signal Strength", f"{signals['signal_strength']}/10")
            
            with col3:
                trend = signals.get('trend', 'NEUTRAL')
                st.markdown(f"### Trend: {trend}")
            
            st.markdown("---")
            
            # Technical indicators tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 RSI", "📊 MACD", "🎯 Support/Resistance", "📉 Moving Averages"])
            
            with tab1:
                st.subheader("Relative Strength Index (RSI)")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # RSI chart
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=tech_data.index,
                        y=tech_data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='#667eea', width=2)
                    ))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.update_layout(template='plotly_dark', height=400)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    current_rsi = tech_data['RSI'].iloc[-1]
                    rsi_gauge = create_gauge_chart(current_rsi, "RSI", 0, 100)
                    st.plotly_chart(rsi_gauge, use_container_width=True)
                    
                    if current_rsi > 70:
                        st.warning("⚠️ Overbought - Consider Selling")
                    elif current_rsi < 30:
                        st.success("✅ Oversold - Consider Buying")
                    else:
                        st.info("ℹ️ Neutral Zone")
            
            with tab2:
                st.subheader("MACD (Moving Average Convergence Divergence)")
                
                # MACD chart
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=tech_data.index,
                    y=tech_data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='#667eea', width=2)
                ))
                fig_macd.add_trace(go.Scatter(
                    x=tech_data.index,
                    y=tech_data['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='orange', width=2)
                ))
                fig_macd.add_trace(go.Bar(
                    x=tech_data.index,
                    y=tech_data['MACD_Hist'],
                    name='Histogram',
                    marker_color='rgba(102, 126, 234, 0.5)'
                ))
                fig_macd.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig_macd, use_container_width=True)
                
                macd_signal = signals.get('macd_signal', 'NEUTRAL')
                if macd_signal == 'BUY':
                    st.success("✅ MACD Crossover - Bullish Signal")
                elif macd_signal == 'SELL':
                    st.error("❌ MACD Crossover - Bearish Signal")
                else:
                    st.info("ℹ️ No Clear MACD Signal")
            
            with tab3:
                st.subheader("Support and Resistance Levels")
                
                # Calculate support and resistance
                support_resistance = components['technical_analyzer'].find_support_resistance(tech_data)
                
                # Price chart with S/R levels
                fig_sr = go.Figure()
                fig_sr.add_trace(go.Candlestick(
                    x=tech_data.index,
                    open=tech_data['Open'],
                    high=tech_data['High'],
                    low=tech_data['Low'],
                    close=tech_data['Close'],
                    name='Price'
                ))
                
                for level in support_resistance['resistance']:
                    fig_sr.add_hline(y=level, line_dash="dash", line_color="red", 
                                    annotation_text=f"R: {level:.2f}")
                
                for level in support_resistance['support']:
                    fig_sr.add_hline(y=level, line_dash="dash", line_color="green", 
                                    annotation_text=f"S: {level:.2f}")
                
                fig_sr.update_layout(template='plotly_dark', height=500)
                st.plotly_chart(fig_sr, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🔴 Resistance Levels")
                    for level in support_resistance['resistance']:
                        st.write(f"${level:.2f}")
                
                with col2:
                    st.markdown("### 🟢 Support Levels")
                    for level in support_resistance['support']:
                        st.write(f"${level:.2f}")
            
            with tab4:
                st.subheader("Moving Averages")
                
                # MA chart
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(
                    x=tech_data.index,
                    y=tech_data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='white', width=2)
                ))
                fig_ma.add_trace(go.Scatter(
                    x=tech_data.index,
                    y=tech_data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1.5)
                ))
                fig_ma.add_trace(go.Scatter(
                    x=tech_data.index,
                    y=tech_data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='blue', width=1.5)
                ))
                fig_ma.add_trace(go.Scatter(
                    x=tech_data.index,
                    y=tech_data['EMA_12'],
                    mode='lines',
                    name='EMA 12',
                    line=dict(color='green', width=1.5)
                ))
                fig_ma.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig_ma, use_container_width=True)
                
                # Moving average signals
                ma_signal = signals.get('ma_signal', 'NEUTRAL')
                if ma_signal == 'BUY':
                    st.success("✅ Price Above Moving Averages - Bullish")
                elif ma_signal == 'SELL':
                    st.error("❌ Price Below Moving Averages - Bearish")
                else:
                    st.info("ℹ️ Mixed Moving Average Signals")
        else:
            st.error("❌ Unable to fetch stock data.")

# Page: ML Predictions
elif page == "🤖 ML Predictions":
    st.title("🤖 Machine Learning Predictions")
    
    stock_symbol = st.text_input("🔎 Enter Stock Symbol", "AAPL")
    
    if stock_symbol:
        with st.spinner("🔄 Training ML models and generating predictions..."):
            stock_data = components['data_fetcher'].get_stock_data(stock_symbol, period='2y')
            
            if stock_data is not None:
                # Train and predict
                predictions = components['ml_predictor'].predict(stock_data, days_ahead=30)
                
                if predictions:
                    st.success("✅ Predictions generated successfully!")
                    
                    # Display prediction gauge
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        trend = predictions['trend']
                        confidence = predictions['confidence']
                        
                        st.markdown(f"### Prediction: **{trend}**")
                        st.markdown(f"Confidence: **{confidence:.1f}%**")
                    
                    with col2:
                        predicted_change = predictions['predicted_change']
                        st.metric("Expected Change", f"{predicted_change:+.2f}%")
                    
                    with col3:
                        accuracy = predictions.get('model_accuracy', 0)
                        st.metric("Model Accuracy", f"{accuracy:.1f}%")
                    
                    st.markdown("---")
                    
                    # Prediction chart
                    st.subheader("📈 Price Prediction (Next 30 Days)")
                    
                    fig_pred = go.Figure()
                    
                    # Historical prices
                    fig_pred.add_trace(go.Scatter(
                        x=stock_data.index[-90:],
                        y=stock_data['Close'][-90:],
                        mode='lines',
                        name='Historical',
                        line=dict(color='#667eea', width=2)
                    ))
                    
                    # Predictions
                    if 'future_dates' in predictions and 'future_prices' in predictions:
                        fig_pred.add_trace(go.Scatter(
                            x=predictions['future_dates'],
                            y=predictions['future_prices'],
                            mode='lines',
                            name='Predicted',
                            line=dict(color='orange', width=2, dash='dash')
                        ))
                        
                        # Confidence interval
                        fig_pred.add_trace(go.Scatter(
                            x=predictions['future_dates'],
                            y=predictions.get('upper_bound', predictions['future_prices']),
                            fill=None,
                            mode='lines',
                            line_color='rgba(255,165,0,0)',
                            showlegend=False
                        ))
                        fig_pred.add_trace(go.Scatter(
                            x=predictions['future_dates'],
                            y=predictions.get('lower_bound', predictions['future_prices']),
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(255,165,0,0)',
                            name='Confidence Interval',
                            fillcolor='rgba(255,165,0,0.2)'
                        ))
                    
                    fig_pred.update_layout(
                        template='plotly_dark',
                        height=500,
                        xaxis_title="Date",
                        yaxis_title="Price ($)"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Feature importance
                    if 'feature_importance' in predictions:
                        st.subheader("🎯 Feature Importance")
                        fig_importance = px.bar(
                            predictions['feature_importance'],
                            x='importance',
                            y='feature',
                            orientation='h',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Model comparison
                    st.subheader("🏆 Model Performance Comparison")
                    if 'model_comparison' in predictions:
                        df_models = pd.DataFrame(predictions['model_comparison'])
                        st.dataframe(df_models, use_container_width=True, hide_index=True)
                else:
                    st.error("❌ Unable to generate predictions.")
            else:
                st.error("❌ Unable to fetch stock data.")

# Page: Sentiment Dashboard
elif page == "💭 Sentiment Dashboard":
    st.title("💭 Advanced Sentiment Analysis Dashboard")
    
    stock_symbol = st.text_input("🔎 Enter Stock Symbol", "AAPL")
    
    if stock_symbol:
        with st.spinner("🔄 Analyzing sentiment from multiple sources..."):
            # Fetch news and analyze sentiment
            news_data = components['data_fetcher'].get_stock_news(stock_symbol, count=50)
            
            if news_data:
                # Analyze sentiment
                sentiment_results = components['sentiment_analyzer'].analyze_news_batch(news_data)
                
                # Overall sentiment gauge
                st.subheader("🎯 Overall Sentiment Score")
                
                col1, col2, col3 = st.columns(3)
                
                overall_score = sentiment_results['overall_sentiment']
                sentiment_label = sentiment_results['sentiment_label']
                
                with col1:
                    gauge = create_gauge_chart(
                        (overall_score + 1) * 50,  # -1 to 1 scale to 0-100
                        "Sentiment",
                        0,
                        100
                    )
                    st.plotly_chart(gauge, use_container_width=True)
                
                with col2:
                    st.markdown(f"### {sentiment_label}")
                    
                    if sentiment_label == "🟢 BULLISH":
                        st.success("Positive market sentiment")
                    elif sentiment_label == "🔴 BEARISH":
                        st.error("Negative market sentiment")
                    else:
                        st.info("Neutral market sentiment")
                    
                    st.metric("Sentiment Score", f"{overall_score:.3f}")
                
                with col3:
                    st.metric("Articles Analyzed", len(news_data))
                    st.metric("Positive", sentiment_results['positive_count'])
                    st.metric("Negative", sentiment_results['negative_count'])
                
                st.markdown("---")
                
                # Sentiment distribution
                st.subheader("📊 Sentiment Distribution")
                
                fig_dist = go.Figure(data=[go.Pie(
                    labels=['Positive', 'Neutral', 'Negative'],
                    values=[
                        sentiment_results['positive_count'],
                        sentiment_results['neutral_count'],
                        sentiment_results['negative_count']
                    ],
                    marker=dict(colors=['#00ff00', '#ffff00', '#ff0000']),
                    hole=0.4
                )])
                fig_dist.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Sentiment timeline
                if 'sentiment_timeline' in sentiment_results:
                    st.subheader("📈 Sentiment Trend Over Time")
                    timeline_df = pd.DataFrame(sentiment_results['sentiment_timeline'])
                    
                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Scatter(
                        x=timeline_df['date'],
                        y=timeline_df['sentiment'],
                        mode='lines+markers',
                        name='Sentiment',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=8)
                    ))
                    fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_timeline.update_layout(
                        template='plotly_dark',
                        height=400,
                        xaxis_title="Date",
                        yaxis_title="Sentiment Score"
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Sentiment clustering
                st.subheader("🔍 Sentiment Clustering Analysis")
                if 'clusters' in sentiment_results:
                    for cluster_id, cluster_data in sentiment_results['clusters'].items():
                        with st.expander(f"📁 Cluster {cluster_id}: {cluster_data['theme']}"):
                            st.markdown(f"**Average Sentiment:** {cluster_data['avg_sentiment']:.3f}")
                            st.markdown(f"**Articles:** {cluster_data['count']}")
                            st.markdown("**Sample Headlines:**")
                            for headline in cluster_data['sample_headlines'][:3]:
                                st.markdown(f"- {headline}")
                
                # Recent news with sentiment
                st.subheader("📰 Recent News & Sentiment")
                
                for idx, article in enumerate(sentiment_results.get('articles', [])[:10]):
                    sentiment = article['sentiment']
                    color = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "🟡"
                    
                    with st.expander(f"{color} {article['title']}"):
                        st.markdown(f"**Sentiment Score:** {sentiment:.3f}")
                        st.markdown(f"**Published:** {article.get('published', 'N/A')}")
                        st.markdown(f"**Summary:** {article.get('summary', 'No summary available')}")
                        if article.get('link'):
                            st.markdown(f"[Read More]({article['link']})")
            else:
                st.warning("⚠️ No news data available for this stock.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. Not financial advice.</p>
    <p>Made with ❤️ using Streamlit, FinBERT, and Advanced ML</p>
</div>
""", unsafe_allow_html=True)
