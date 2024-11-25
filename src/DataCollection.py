import yfinance as yf
from src.DataProcessor import DataProcessor

class DataCollection:
    @staticmethod
    def get_asset_data(symbol: str, asset_type: str):
        if not symbol:
            return "Error: Symbol cannot be empty"
            
        try:
            # Add '-USD' suffix for crypto symbols
            ticker_symbol = f"{symbol}-USD" if asset_type.lower() == "crypto" else symbol
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # Verify that we got valid data
            if not info:
                return "Error: No data found for this symbol"
            
            # Process the data using DataProcessor
            processed_data = DataProcessor.process_asset_data(info, asset_type, ticker)
            if not processed_data:
                return "Error: Could not process asset data"
            
            # Get historical data and training sets
            historical_data = DataProcessor.process_historical_data(ticker, asset_type)
            if isinstance(historical_data, dict):
                processed_data['training_sets'] = historical_data['training_sets']
                processed_data['historical_data'] = historical_data['historical_data']
            
            return processed_data
        except Exception as e:
            return f"Error fetching data: {str(e)}"

