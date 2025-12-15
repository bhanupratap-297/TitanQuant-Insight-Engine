"""
Machine Learning Predictor Module
ml_predictor.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class MLPredictor:
    def __init__(self):
        """Initialize ML models"""
        self.scaler = MinMaxScaler()
        self.models = {}
        self.feature_names = []
    
    def create_features(self, data):
        """
        Create features for ML models
        """
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Price momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        # Rate of change
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Price position relative to moving averages
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
        
        # High-Low range
        df['HL_Range'] = df['High'] - df['Low']
        df['HL_Pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Target variable (next day's price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest model
        """
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        return {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'accuracy': max(0, test_r2 * 100),
            'feature_importance': dict(zip(self.feature_names, model.feature_importances_))
        }
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """
        Train Gradient Boosting model
        """
        model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        return {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'accuracy': max(0, test_r2 * 100),
            'feature_importance': dict(zip(self.feature_names, model.feature_importances_))
        }
    
    def predict(self, data, days_ahead=30):
        """
        Make predictions using ensemble of models
        """
        try:
            # Create features
            df_features = self.create_features(data)
            
            # Prepare features and target
            feature_cols = [col for col in df_features.columns if col not in 
                          ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            self.feature_names = feature_cols
            X = df_features[feature_cols].values
            y = df_features['Target'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Train models
            rf_results = self.train_random_forest(X_train, y_train, X_test, y_test)
            gb_results = self.train_gradient_boosting(X_train, y_train, X_test, y_test)
            
            # Ensemble prediction
            last_features = X[-1].reshape(1, -1)
            rf_pred = rf_results['model'].predict(last_features)[0]
            gb_pred = gb_results['model'].predict(last_features)[0]
            
            # Average predictions
            ensemble_pred = (rf_pred + gb_pred) / 2
            
            current_price = data['Close'].iloc[-1]
            predicted_change = ((ensemble_pred - current_price) / current_price) * 100
            
            # Determine trend
            if predicted_change > 2:
                trend = "📈 STRONG BUY"
                confidence = min(95, 70 + abs(predicted_change))
            elif predicted_change > 0.5:
                trend = "📊 BUY"
                confidence = min(85, 60 + abs(predicted_change))
            elif predicted_change < -2:
                trend = "📉 STRONG SELL"
                confidence = min(95, 70 + abs(predicted_change))
            elif predicted_change < -0.5:
                trend = "📊 SELL"
                confidence = min(85, 60 + abs(predicted_change))
            else:
                trend = "➡️ HOLD"
                confidence = 50
            
            # Generate future predictions
            future_dates = pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=days_ahead
            )
            
            # Simple projection based on predicted trend
            future_prices = [current_price]
            daily_change = predicted_change / days_ahead / 100
            
            for i in range(days_ahead):
                next_price = future_prices[-1] * (1 + daily_change)
                future_prices.append(next_price)
            
            future_prices = future_prices[1:]
            
            # Confidence intervals
            volatility = data['Close'].pct_change().std()
            upper_bound = [p * (1 + volatility * 2) for p in future_prices]
            lower_bound = [p * (1 - volatility * 2) for p in future_prices]
            
            # Feature importance
            feature_importance_df = pd.DataFrame([
                {'feature': k, 'importance': v} 
                for k, v in rf_results['feature_importance'].items()
            ]).sort_values('importance', ascending=False).head(10)
            
            # Model comparison
            model_comparison = [
                {
                    'Model': 'Random Forest',
                    'Accuracy': f"{rf_results['accuracy']:.2f}%",
                    'R² Score': f"{rf_results['test_r2']:.4f}",
                    'RMSE': f"{np.sqrt(rf_results['test_mse']):.2f}"
                },
                {
                    'Model': 'Gradient Boosting',
                    'Accuracy': f"{gb_results['accuracy']:.2f}%",
                    'R² Score': f"{gb_results['test_r2']:.4f}",
                    'RMSE': f"{np.sqrt(gb_results['test_mse']):.2f}"
                }
            ]
            
            return {
                'trend': trend,
                'confidence': confidence,
                'predicted_change': predicted_change,
                'current_price': current_price,
                'predicted_price': ensemble_pred,
                'future_dates': future_dates.tolist(),
                'future_prices': future_prices,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'model_accuracy': (rf_results['accuracy'] + gb_results['accuracy']) / 2,
                'feature_importance': feature_importance_df.to_dict('records'),
                'model_comparison': model_comparison
            }
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
