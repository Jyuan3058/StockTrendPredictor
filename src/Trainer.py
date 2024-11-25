from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

class Trainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.backtest_results = None

    def train(self, training_sets):
        """Train the model using the provided training sets"""
        try:
            # Get all features and labels
            X = pd.DataFrame(training_sets['all']['features'], columns=[
                'SMA_50_ratio',
                'SMA_200_ratio',
                'RSI',
                'MACD',
                'MACD_Signal',
                'Volatility',
                'BB_Position'
            ])
            y = training_sets['all']['labels']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale the features while preserving column names
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

            # Train the model
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True

            # Calculate accuracy
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test)

            # Store feature importance
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)

            return {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'feature_importance': self.feature_importance.to_dict()
            }

        except Exception as e:
            return f"Error training model: {str(e)}"

    def predict(self, features):
        """Make predictions using the trained model"""
        if not self.is_trained:
            return "Error: Model not trained yet"

        try:
            # Convert to DataFrame with feature names
            features_df = pd.DataFrame(features, columns=[
                'SMA_50_ratio',
                'SMA_200_ratio',
                'RSI',
                'MACD',
                'MACD_Signal',
                'Volatility',
                'BB_Position'
            ])
            
            # Scale the features
            features_scaled = pd.DataFrame(
                self.scaler.transform(features_df),
                columns=features_df.columns
            )
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features_scaled)
            # Get predicted class
            prediction = self.model.predict(features_scaled)

            return {
                'prediction': prediction[0],
                'probability': probabilities[0][prediction[0]],
                'feature_values': features_df.iloc[0].to_dict()  # Include actual feature values
            }

        except Exception as e:
            return f"Error predicting: {str(e)}"

    def backtest(self, historical_data: pd.DataFrame, initial_capital: float = 10000.0, window: int = 20):
        """
        Perform backtesting of the model on historical data
        """
        if not self.is_trained:
            return "Error: Model must be trained before backtesting"

        try:
            # Initialize tracking variables
            capital = initial_capital
            position = None
            trades = []
            equity_curve = []
            total_pnl = 0
            trade_durations = []
            winning_trades = []
            losing_trades = []
            
            # Create features for each trading day
            for i in range(window, len(historical_data) - window):
                current_date = historical_data.index[i]
                current_price = historical_data['Close'].iloc[i]
                
                # Create feature set
                feature_set = [{
                    'SMA_50_ratio': historical_data['SMA_50'].iloc[i] / current_price,
                    'SMA_200_ratio': historical_data['SMA_200'].iloc[i] / current_price,
                    'RSI': historical_data['RSI'].iloc[i],
                    'MACD': historical_data['MACD'].iloc[i],
                    'MACD_Signal': historical_data['Signal_Line'].iloc[i],
                    'Volatility': historical_data['Volatility'].iloc[i],
                    'BB_Position': (current_price - historical_data['BB_lower'].iloc[i]) / 
                                 (historical_data['BB_upper'].iloc[i] - historical_data['BB_lower'].iloc[i])
                }]
                
                # Get prediction
                prediction = self.predict(pd.DataFrame(feature_set))
                if isinstance(prediction, str):
                    continue
                
                # Trading logic with enhanced information
                if prediction['prediction'] == 1 and position is None:  # Buy signal
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'entry_date': current_date,
                        'size': capital / current_price,
                        'indicators': {
                            'RSI': historical_data['RSI'].iloc[i],
                            'MACD': historical_data['MACD'].iloc[i],
                            'BB_Position': (current_price - historical_data['BB_lower'].iloc[i]) / 
                                         (historical_data['BB_upper'].iloc[i] - historical_data['BB_lower'].iloc[i])
                        }
                    }
                    trades.append({
                        'date': current_date,
                        'action': 'buy',
                        'price': current_price,
                        'confidence': prediction['probability'],
                        'indicators': position['indicators']
                    })
                    
                elif prediction['prediction'] == 0 and position is not None:  # Sell signal
                    # Calculate trade metrics
                    pnl = (current_price - position['entry_price']) * position['size']
                    duration = (current_date - position['entry_date']).days
                    trade_durations.append(duration)
                    total_pnl += pnl
                    
                    trade_result = {
                        'date': current_date,
                        'action': 'sell',
                        'price': current_price,
                        'confidence': prediction['probability'],
                        'pnl': pnl,
                        'return_pct': (pnl / (position['entry_price'] * position['size'])) * 100,
                        'duration_days': duration,
                        'exit_indicators': {
                            'RSI': historical_data['RSI'].iloc[i],
                            'MACD': historical_data['MACD'].iloc[i],
                            'BB_Position': (current_price - historical_data['BB_lower'].iloc[i]) / 
                                         (historical_data['BB_upper'].iloc[i] - historical_data['BB_lower'].iloc[i])
                        }
                    }
                    
                    trades.append(trade_result)
                    
                    # Track winning/losing trades
                    if pnl > 0:
                        winning_trades.append(trade_result)
                    else:
                        losing_trades.append(trade_result)
                    
                    capital += pnl
                    position = None
                
                # Track equity
                current_equity = capital
                if position is not None:
                    current_equity = capital + (current_price - position['entry_price']) * position['size']
                equity_curve.append({
                    'date': current_date,
                    'equity': current_equity
                })
            
            # Close any open position at the end
            if position is not None:
                final_price = historical_data['Close'].iloc[-1]
                pnl = (final_price - position['entry_price']) * position['size']
                duration = (historical_data.index[-1] - position['entry_date']).days
                trade_durations.append(duration)
                capital += pnl
                
                trade_result = {
                    'date': historical_data.index[-1],
                    'action': 'sell',
                    'price': final_price,
                    'pnl': pnl,
                    'return_pct': (pnl / (position['entry_price'] * position['size'])) * 100,
                    'duration_days': duration
                }
                
                trades.append(trade_result)
                
                if pnl > 0:
                    winning_trades.append(trade_result)
                else:
                    losing_trades.append(trade_result)
            
            # Calculate enhanced performance metrics
            equity_df = pd.DataFrame(equity_curve)
            returns = equity_df['equity'].pct_change().dropna()
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            self.backtest_results = {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': (capital - initial_capital) / initial_capital * 100,
                'total_pnl': total_pnl,
                'number_of_trades': len(trades) // 2,  # Divide by 2 since each trade has buy and sell
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'avg_trade_duration': np.mean(trade_durations) if trade_durations else 0,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win/avg_loss) if avg_loss != 0 else float('inf'),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(equity_df['equity']),
                'trades': trades,
                'equity_curve': equity_curve
            }
            
            return self.backtest_results

        except Exception as e:
            return f"Error during backtesting: {str(e)}"
    
    def _calculate_max_drawdown(self, equity_series):
        """Calculate the maximum drawdown from peak equity"""
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        return drawdown.min() * 100  # Convert to percentage
