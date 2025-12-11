# core/technicals.py
import pandas as pd
import pandas_ta as ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    df with columns: Open, High, Low, Close, Volume (yfinance format).
    Adds MACD, RSI, and simple trend flags.
    """
    df = df.copy()

    # RSI
    df["RSI14"] = ta.rsi(df["Close"], length=14)

    # MACD (12, 26, 9)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)  # adds MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

    # Moving averages for trend direction
    df["SMA50"] = ta.sma(df["Close"], length=50)
    df["SMA200"] = ta.sma(df["Close"], length=200)

    # Simple trend label
    df["Trend"] = "Sideways"
    df.loc[df["SMA50"] > df["SMA200"], "Trend"] = "Uptrend"
    df.loc[df["SMA50"] < df["SMA200"], "Trend"] = "Downtrend"

    return df

def simple_support_resistance(df: pd.DataFrame, window: int = 20):
    """
    Naive support/resistance via rolling min/max of closes.
    """
    s = df["Close"].rolling(window=window).min().iloc[-1]
    r = df["Close"].rolling(window=window).max().iloc[-1]
    return float(s), float(r)


import plotly.graph_objects as go
from core.technicals import add_indicators, simple_support_resistance

stock_data = yf.download(ticker, start=start_date, end=end_date)

if stock_data.empty:
    st.warning("No stock data for selected range.")
else:
    stock_data = add_indicators(stock_data)
    support, resistance = simple_support_resistance(stock_data)

    st.markdown(f"**Support (last {20} days):** {support:.2f}")
    st.markdown(f"**Resistance (last {20} days):** {resistance:.2f}")
    st.markdown(f"**Trend:** {stock_data['Trend'].iloc[-1]}")

    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data["Open"],
        high=stock_data["High"],
        low=stock_data["Low"],
        close=stock_data["Close"],
        name="Price"
    ))

    # Moving averages
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data["SMA50"],
        name="SMA 50"
    ))
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data["SMA200"],
        name="SMA 200"
    ))

    st.plotly_chart(fig, use_container_width=True)

    # RSI + MACD in tabs
    tab_rsi, tab_macd = st.tabs(["RSI", "MACD"])

    with tab_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data["RSI14"],
            name="RSI 14"
        ))
        fig_rsi.add_hrect(y0=30, y1=70, opacity=0.1, line_width=0, fillcolor="gray")
        st.plotly_chart(fig_rsi, use_container_width=True)

    with tab_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data["MACD_12_26_9"],
            name="MACD"
        ))
        fig_macd.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data["MACDs_12_26_9"],
            name="Signal"
        ))
        st.plotly_chart(fig_macd, use_container_width=True)
