"""
Enhanced Technical Indicators Module
enhanced_technical_indicators.py
Includes: Ichimoku, Supertrend, MFI, CCI, Williams %R, Fibonacci, Pivot Points, Volume Profile, Options Greeks
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta

class EnhancedTechnicalIndicators:
    def __init__(self):
        """Initialize enhanced technical indicators"""
        pass
    
    # ============ ICHIMOKU CLOUD ============
    def calculate_ichimoku(self, data, tenkan=9, kijun=26, senkou_b=52, kumo_offset=26):
        """Calculate Ichimoku Cloud indicator
        Returns: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        """
        # Tenkan-sen (Conversion Line)
        high_9 = data['High'].rolling(window=tenkan).max()
        low_9 = data['Low'].rolling(window=tenkan).min()
        tenkan_sen = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = data['High'].rolling(window=kijun).max()
        low_26 = data['Low'].rolling(window=kijun).min()
        kijun_sen = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kumo_offset)
        
        # Senkou Span B (Leading Span B)
        high_52 = data['High'].rolling(window=senkou_b).max()
        low_52 = data['Low'].rolling(window=senkou_b).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(kumo_offset)
        
        # Chikou Span (Lagging Span)
        chikou_span = data['Close'].shift(-kumo_offset)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    # ============ SUPERTREND ============
    def calculate_supertrend(self, data, period=10, multiplier=3):
        """Calculate Supertrend indicator
        Returns: supertrend, trend, direction
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        hl2 = (high + low) / 2
        atr = self._calculate_atr(data, period)
        
        matr = hl2 + multiplier * atr
        mitr = hl2 - multiplier * atr
        
        supertrend = pd.Series(index=data.index, dtype='float64')
        trend = pd.Series(index=data.index, dtype='int64')
        direction = pd.Series(index=data.index, dtype='int64')
        
        supertrend.iloc[0] = matr.iloc[0]
        trend.iloc[0] = 1
        direction.iloc[0] = 1
        
        for i in range(1, len(data)):
            if matr.iloc[i] < supertrend.iloc[i-1] or close.iloc[i-1] > supertrend.iloc[i-1]:
                matr.iloc[i] = matr.iloc[i]
            else:
                matr.iloc[i] = supertrend.iloc[i-1]
            
            if mitr.iloc[i] > supertrend.iloc[i-1] or close.iloc[i-1] < supertrend.iloc[i-1]:
                mitr.iloc[i] = mitr.iloc[i]
            else:
                mitr.iloc[i] = supertrend.iloc[i-1]
            
            if supertrend.iloc[i-1] == matr.iloc[i-1]:
                if close.iloc[i] <= matr.iloc[i]:
                    supertrend.iloc[i] = matr.iloc[i]
                    trend.iloc[i] = 1
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = mitr.iloc[i]
                    trend.iloc[i] = 0
                    direction.iloc[i] = -1
            else:
                if close.iloc[i] >= mitr.iloc[i]:
                    supertrend.iloc[i] = mitr.iloc[i]
                    trend.iloc[i] = 0
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = matr.iloc[i]
                    trend.iloc[i] = 1
                    direction.iloc[i] = 1
        
        return supertrend, trend, direction
    
    # ============ MONEY FLOW INDEX (MFI) ============
    def calculate_mfi(self, data, period=14):
        """Calculate Money Flow Index (MFI)"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = pd.Series(index=data.index, dtype='float64')
        negative_flow = pd.Series(index=data.index, dtype='float64')
        
        for i in range(len(data)):
            if i == 0:
                positive_flow.iloc[i] = 0
                negative_flow.iloc[i] = 0
            else:
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.iloc[i] = money_flow.iloc[i]
                    negative_flow.iloc[i] = 0
                else:
                    positive_flow.iloc[i] = 0
                    negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_flow_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    # ============ COMMODITY CHANNEL INDEX (CCI) ============
    def calculate_cci(self, data, period=20):
        """Calculate Commodity Channel Index (CCI)"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        cci = (typical_price - sma) / (0.015 * mad)
        
        return cci
    
    # ============ WILLIAMS %R ============
    def calculate_williams_r(self, data, period=14):
        """Calculate Williams %R indicator"""
        high = data['High'].rolling(window=period).max()
        low = data['Low'].rolling(window=period).min()
        
        williams_r = -100 * ((high - data['Close']) / (high - low))
        
        return williams_r
    
    # ============ FIBONACCI RETRACEMENT ============
    def calculate_fibonacci_levels(self, data):
        """Calculate Fibonacci retracement levels"""
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        
        levels = {
            '0%': high,
            '23.6%': high - (diff * 0.236),
            '38.2%': high - (diff * 0.382),
            '50%': high - (diff * 0.5),
            '61.8%': high - (diff * 0.618),
            '78.6%': high - (diff * 0.786),
            '100%': low
        }
        
        return levels
    
    # ============ PIVOT POINTS ============
    def calculate_pivot_points(self, data):
        """Calculate Daily Pivot Points"""
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        
        return {
            'pivot': pivot,
            'resistance_1': r1,
            'resistance_2': r2,
            'support_1': s1,
            'support_2': s2
        }
    
    # ============ VOLUME PROFILE ============
    def calculate_volume_profile(self, data, bins=20):
        """Calculate Volume Profile"""
        price_range = np.linspace(data['Low'].min(), data['High'].max(), bins)
        volume_profile = pd.cut(data['Close'], bins=price_range).value_counts().sort_index()
        
        return volume_profile
    
    # ============ OPTIONS GREEKS ============
    def calculate_option_greeks(self, spot_price, strike_price, time_to_expiry, 
                                risk_free_rate=0.05, volatility=0.2, option_type='call'):
        """Calculate Options Greeks using Black-Scholes model"""
        d1 = (np.log(spot_price / strike_price) + 
              (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) - 
                    risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
        else:  # put
            delta = norm.cdf(d1) - 1
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) + 
                    risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 365
        
        gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))
        vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100
        rho = (strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * 
               norm.cdf(d2) if option_type == 'call' else 
               -strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    # ============ HELPER METHODS ============
    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range (ATR) - helper method"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def get_comprehensive_analysis(self, data):
        """Get comprehensive technical analysis with all indicators""" 
        analysis = {}
        
        # Ichimoku
        ichimoku = self.calculate_ichimoku(data)
        analysis['ichimoku'] = ichimoku
        
        # Supertrend
        supertrend, trend, direction = self.calculate_supertrend(data)
        analysis['supertrend'] = supertrend
        analysis['supertrend_trend'] = trend
        analysis['supertrend_direction'] = direction
        
        # Money Flow Index
        analysis['mfi'] = self.calculate_mfi(data)
        
        # CCI
        analysis['cci'] = self.calculate_cci(data)
        
        # Williams %R
        analysis['williams_r'] = self.calculate_williams_r(data)
        
        # Fibonacci
        analysis['fibonacci'] = self.calculate_fibonacci_levels(data)
        
        # Pivot Points
        analysis['pivot_points'] = self.calculate_pivot_points(data)
        
        # Options Greeks (for current price)
        current_price = data['Close'].iloc[-1]
        analysis['options_greeks'] = self.calculate_option_greeks(current_price, current_price)
        
        return analysis
