from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class DataProcessor:
    STOCK_FIELDS = {
        'longName': 'Name',
        'regularMarketPrice': 'Current Price',
        'regularMarketChangePercent': 'Change (%)',
        'fiftyDayAverage': '50 Day Avg',
        'twoHundredDayAverage': '200 Day Avg',
        'regularMarketVolume': 'Volume',
        'averageVolume': 'Avg Volume',
        'marketCap': 'Market Cap',
        'beta': 'Beta',
        'forwardPE': 'Forward P/E',
    }
    
    CRYPTO_FIELDS = {
        'longName': 'Name',
        'regularMarketPrice': 'Current Price',
        'regularMarketChangePercent': 'Change (%)',
        'volume24Hr': '24h Volume',
        'marketCap': 'Market Cap',
        'circulatingSupply': 'Circulating Supply',
        'totalSupply': 'Total Supply',
    }

    @staticmethod
    def process_historical_data(ticker, asset_type: str):
        """
        Fetch and process historical data for prediction
        Returns a dictionary containing processed data and technical indicators
        """
        try:
            # Get maximum available historical data
            hist = ticker.history(period="max")
            
            if hist.empty:
                return "Error: No historical data available"

            # Ensure all necessary fields exist for backtesting
            hist['Close'] = hist['Close'].fillna(method='ffill')
            hist['Open'] = hist['Open'].fillna(hist['Close'])
            hist['High'] = hist['High'].fillna(hist['Close'])
            hist['Low'] = hist['Low'].fillna(hist['Close'])
            hist['Volume'] = hist['Volume'].fillna(0)
            
            # Calculate technical indicators
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            
            # Calculate RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
            hist['MACD'] = exp1 - exp2
            hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
            
            # Calculate Bollinger Bands
            hist['BB_middle'] = hist['Close'].rolling(window=20).mean()
            hist['BB_upper'] = hist['BB_middle'] + 2 * hist['Close'].rolling(window=20).std()
            hist['BB_lower'] = hist['BB_middle'] - 2 * hist['Close'].rolling(window=20).std()
            
            # Calculate daily returns and volatility
            hist['Returns'] = hist['Close'].pct_change()
            hist['Volatility'] = hist['Returns'].rolling(window=21).std() * np.sqrt(252)

            # Get the most recent data
            latest_data = hist.tail(1)
            
            # Add training sets to the return dictionary
            training_sets = DataProcessor.create_training_sets(hist)
            
            return {
                'historical_data': hist,
                'latest_indicators': {
                    'Last Close': latest_data['Close'].iloc[-1],
                    'SMA 50': latest_data['SMA_50'].iloc[-1],
                    'SMA 200': latest_data['SMA_200'].iloc[-1],
                    'RSI': latest_data['RSI'].iloc[-1],
                    'MACD': latest_data['MACD'].iloc[-1],
                    'Signal Line': latest_data['Signal_Line'].iloc[-1],
                    'Volatility': latest_data['Volatility'].iloc[-1],
                    'BB Upper': latest_data['BB_upper'].iloc[-1],
                    'BB Middle': latest_data['BB_middle'].iloc[-1],
                    'BB Lower': latest_data['BB_lower'].iloc[-1],
                },
                'training_sets': training_sets
            }
        except Exception as e:
            return f"Error processing historical data: {str(e)}"

    @staticmethod
    def format_number(value):
        """Format numbers with appropriate notation"""
        try:
            if isinstance(value, (int, float)):
                if value >= 1_000_000_000:  # Billions
                    return f"{value/1_000_000_000:.2f}B"
                elif value >= 1_000_000:  # Millions
                    return f"{value/1_000_000:.2f}M"
                elif value >= 1000:
                    return f"{value:,.2f}"
                elif value >= 1:
                    return f"{value:.2f}"
                else:
                    return f"{value:.6f}"
            return str(value)
        except (TypeError, ValueError):
            return "N/A"

    @staticmethod
    def process_asset_data(data: dict, asset_type: str, ticker):
        """Process and clean the raw asset data"""
        if not isinstance(data, dict):
            return {}

        # Select relevant fields based on asset type
        fields = DataProcessor.CRYPTO_FIELDS if asset_type.lower() == "crypto" else DataProcessor.STOCK_FIELDS
        
        # Process current market data
        processed_data = {
            new_key: DataProcessor.format_number(data.get(old_key, "N/A"))
            for old_key, new_key in fields.items()
        }
        
        # Get historical data and technical indicators
        historical_data = DataProcessor.process_historical_data(ticker, asset_type)
        
        if isinstance(historical_data, str):  # Error occurred
            return historical_data
            
        # Add technical indicators to processed data
        processed_data.update({
            key: DataProcessor.format_number(value)
            for key, value in historical_data['latest_indicators'].items()
        })
        
        return {
            'market_data': processed_data,
            'historical_data': historical_data['historical_data']
        }

    @staticmethod
    def identify_trends(historical_data: pd.DataFrame, window: int = 20, threshold: float = 0.02):
        """
        Identify positive and negative trends in the price data
        
        Args:
            historical_data: DataFrame with historical price data
            window: Number of days to consider for trend (default 20 trading days)
            threshold: Minimum price change to consider as trend (default 2%)
        
        Returns:
            DataFrame with trend labels
        """
        # Calculate percentage change over the window period
        data = historical_data.copy()
        data['Price_Change'] = (data['Close'].shift(-window) - data['Close']) / data['Close']
        
        # Create trend labels
        data['Trend'] = 'neutral'
        data.loc[data['Price_Change'] >= threshold, 'Trend'] = 'positive'
        data.loc[data['Price_Change'] <= -threshold, 'Trend'] = 'negative'
        
        return data

    @staticmethod
    def create_training_sets(historical_data: pd.DataFrame, window: int = 20, threshold: float = 0.02):
        """
        Create training sets for both positive and negative trends
        
        Args:
            historical_data: DataFrame with historical price data
            window: Number of days to look back for features
            threshold: Minimum price change to consider as trend
        
        Returns:
            Dictionary containing training features and labels for both trends
        """
        # Get trend data
        trend_data = DataProcessor.identify_trends(historical_data, window, threshold)
        
        # Create features and labels
        features = []
        labels = []
        
        for i in range(window, len(trend_data) - window):
            # Skip if trend is neutral
            if trend_data['Trend'].iloc[i] == 'neutral':
                continue
                
            # Create feature set from technical indicators
            feature_set = {
                'SMA_50_ratio': trend_data['SMA_50'].iloc[i] / trend_data['Close'].iloc[i],
                'SMA_200_ratio': trend_data['SMA_200'].iloc[i] / trend_data['Close'].iloc[i],
                'RSI': trend_data['RSI'].iloc[i],
                'MACD': trend_data['MACD'].iloc[i],
                'MACD_Signal': trend_data['Signal_Line'].iloc[i],
                'Volatility': trend_data['Volatility'].iloc[i],
                'BB_Position': (trend_data['Close'].iloc[i] - trend_data['BB_lower'].iloc[i]) / 
                              (trend_data['BB_upper'].iloc[i] - trend_data['BB_lower'].iloc[i])
            }
            
            features.append(feature_set)
            labels.append(1 if trend_data['Trend'].iloc[i] == 'positive' else 0)
        
        # Convert to numpy arrays
        X = pd.DataFrame(features).to_numpy()
        y = np.array(labels)
        
        # Separate positive and negative trends
        positive_mask = y == 1
        negative_mask = y == 0
        
        return {
            'positive': {
                'features': X[positive_mask],
                'labels': y[positive_mask]
            },
            'negative': {
                'features': X[negative_mask],
                'labels': y[negative_mask]
            },
            'all': {
                'features': X,
                'labels': y
            }
        }
