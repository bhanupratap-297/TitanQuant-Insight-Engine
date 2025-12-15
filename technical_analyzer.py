
"""
Technical Analysis Module
technical_analyzer.py
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


class TechnicalAnalyzer:
    def __init__(self):
        """Initialize technical analyzer"""
        pass
    
    def calculate_rsi(self, data, period=14):
        """
        Calculate Relative Strength Index (RSI)
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        """
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def calculate_bollinger_bands(self, data, period=20, std=2):
        """
        Calculate Bollinger Bands
        """
        sma = data['Close'].rolling(window=period).mean()
        rolling_std = data['Close'].rolling(window=period).std()
        
        upper_band = sma + (rolling_std * std)
        lower_band = sma - (rolling_std * std)
        
        return upper_band, sma, lower_band
    
    def calculate_moving_averages(self, data):
        """
        Calculate various moving averages
        """
        ma_dict = {
            'SMA_20': data['Close'].rolling(window=20).mean(),
            'SMA_50': data['Close'].rolling(window=50).mean(),
            'SMA_200': data['Close'].rolling(window=200).mean(),
            'EMA_12': data['Close'].ewm(span=12, adjust=False).mean(),
            'EMA_26': data['Close'].ewm(span=26, adjust=False).mean(),
            'EMA_50': data['Close'].ewm(span=50, adjust=False).mean()
        }
        
        return ma_dict
    
    def calculate_stochastic(self, data, period=14):
        """
        Calculate Stochastic Oscillator
        """
        low_min = data['Low'].rolling(window=period).min()
        high_max = data['High'].rolling(window=period).max()
        
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent, d_percent
    
    def calculate_atr(self, data, period=14):
        """
        Calculate Average True Range (ATR)
        """
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def calculate_obv(self, data):
        """
        Calculate On-Balance Volume (OBV)
        """
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        return obv
    
    def calculate_adx(self, data, period=14):
        """
        Calculate Average Directional Index (ADX)
        """
        # Calculate +DM and -DM
        high_diff = data['High'].diff()
        low_diff = -data['Low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR
        atr = self.calculate_atr(data, period)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def find_support_resistance(self, data, order=5):
        """
        Find support and resistance levels using local extrema
        """
        # Find local maxima (resistance)
        highs = data['High'].values
        resistance_idx = argrelextrema(highs, np.greater, order=order)[0]
        resistance_levels = sorted(set(highs[resistance_idx]), reverse=True)[:3]
        
        # Find local minima (support)
        lows = data['Low'].values
        support_idx = argrelextrema(lows, np.less, order=order)[0]
        support_levels = sorted(set(lows[support_idx]))[:3]
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def calculate_all_indicators(self, data):
        """
        Calculate all technical indicators
        """
        df = data.copy()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df)
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df)
        
        # Moving Averages
        ma_dict = self.calculate_moving_averages(df)
        for key, value in ma_dict.items():
            df[key] = value
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(df)
        
        # ATR
        df['ATR'] = self.calculate_atr(df)
        
        # OBV
        df['OBV'] = self.calculate_obv(df)
        
        # ADX
        df['ADX'], df['Plus_DI'], df['Minus_DI'] = self.calculate_adx(df)
        
        return df
    
    def generate_signals(self, data):
        """
        Generate trading signals based on technical indicators
        """
        signals = {}
        
        # RSI Signal
        rsi_current = data['RSI'].iloc[-1]
        if rsi_current > 70:
            signals['rsi_signal'] = 'SELL'
            signals['rsi_strength'] = min((rsi_current - 70) / 10, 3)
        elif rsi_current < 30:
            signals['rsi_signal'] = 'BUY'
            signals['rsi_strength'] = min((30 - rsi_current) / 10, 3)
        else:
            signals['rsi_signal'] = 'NEUTRAL'
            signals['rsi_strength'] = 0
        
        # MACD Signal
        macd_current = data['MACD'].iloc[-1]
        macd_signal_current = data['MACD_Signal'].iloc[-1]
        macd_prev = data['MACD'].iloc[-2]
        macd_signal_prev = data['MACD_Signal'].iloc[-2]
        
        if macd_current > macd_signal_current and macd_prev <= macd_signal_prev:
            signals['macd_signal'] = 'BUY'
            signals['macd_strength'] = 2
        elif macd_current < macd_signal_current and macd_prev >= macd_signal_prev:
            signals['macd_signal'] = 'SELL'
            signals['macd_strength'] = 2
        else:
            signals['macd_signal'] = 'NEUTRAL'
            signals['macd_strength'] = 0
        
        # Moving Average Signal
        price_current = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        
        if price_current > sma_20 and sma_20 > sma_50:
            signals['ma_signal'] = 'BUY'
            signals['ma_strength'] = 2
        elif price_current < sma_20 and sma_20 < sma_50:
            signals['ma_signal'] = 'SELL'
            signals['ma_strength'] = 2
        else:
            signals['ma_signal'] = 'NEUTRAL'
            signals['ma_strength'] = 0
        
        # Bollinger Bands Signal
        bb_upper = data['BB_Upper'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        
        if price_current >= bb_upper:
            signals['bb_signal'] = 'SELL'
            signals['bb_strength'] = 1
        elif price_current <= bb_lower:
            signals['bb_signal'] = 'BUY'
            signals['bb_strength'] = 1
        else:
            signals['bb_signal'] = 'NEUTRAL'
            signals['bb_strength'] = 0
        
        # Stochastic Signal
        stoch_k = data['Stoch_K'].iloc[-1]
        if stoch_k > 80:
            signals['stoch_signal'] = 'SELL'
            signals['stoch_strength'] = 1
        elif stoch_k < 20:
            signals['stoch_signal'] = 'BUY'
            signals['stoch_strength'] = 1
        else:
            signals['stoch_signal'] = 'NEUTRAL'
            signals['stoch_strength'] = 0
        
        # Calculate overall signal
        buy_strength = 0
        sell_strength = 0
        
        for key, value in signals.items():
            if '_strength' in key:
                signal_type = key.replace('_strength', '_signal')
                if signals.get(signal_type) == 'BUY':
                    buy_strength += value
                elif signals.get(signal_type) == 'SELL':
                    sell_strength += value
        
        total_strength = buy_strength + sell_strength
        if total_strength > 0:
            signal_strength = max(buy_strength, sell_strength)
        else:
            signal_strength = 0
        
        if buy_strength > sell_strength and buy_strength >= 3:
            overall_signal = 'BUY'
        elif sell_strength > buy_strength and sell_strength >= 3:
            overall_signal = 'SELL'
        else:
            overall_signal = 'HOLD'
        
        # Determine trend
        if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1]:
            trend = 'UPTREND'
        elif data['SMA_20'].iloc[-1] < data['SMA_50'].iloc[-1]:
            trend = 'DOWNTREND'
        else:
            trend = 'SIDEWAYS'
        
        signals['overall_signal'] = overall_signal
        signals['signal_strength'] = min(signal_strength, 10)
        signals['trend'] = trend
        signals['buy_strength'] = buy_strength
        signals['sell_strength'] = sell_strength
        
        return signals
