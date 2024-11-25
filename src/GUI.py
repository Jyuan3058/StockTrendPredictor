import matplotlib
matplotlib.use('Agg')

import tkinter as tk
from tkinter import ttk, messagebox
from src.DataCollection import DataCollection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.Trainer import Trainer
import pandas as pd
from tkinter import scrolledtext
from datetime import datetime

class StockCryptoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock/Crypto Selector")
        self.root.geometry("400x300")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Asset type selection
        self.asset_type = tk.StringVar(value="stock")
        ttk.Label(self.main_frame, text="Select Asset Type:").grid(row=0, column=0, pady=10)
        ttk.Radiobutton(self.main_frame, text="Stock", variable=self.asset_type, value="stock").grid(row=0, column=1)
        ttk.Radiobutton(self.main_frame, text="Crypto", variable=self.asset_type, value="crypto").grid(row=0, column=2)
        
        # Symbol input
        ttk.Label(self.main_frame, text="Enter Symbol:").grid(row=1, column=0, pady=10)
        self.symbol_entry = ttk.Entry(self.main_frame)
        self.symbol_entry.grid(row=1, column=1, columnspan=2, sticky=tk.W+tk.E)
        
        # Submit button
        ttk.Button(self.main_frame, text="Submit", command=self.submit).grid(row=2, column=1, pady=20)
        
        self.trainer = Trainer()
        self.backtest_results = None
        
    def create_trade_details_window(self, trades):
        """Create a new window to display detailed trade information"""
        details_window = tk.Toplevel(self.root)
        details_window.title("Trade Details")
        details_window.geometry("800x600")

        # Create text widget with scrollbar
        text_widget = scrolledtext.ScrolledText(details_window, wrap=tk.WORD, width=80, height=30)
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Header
        text_widget.insert(tk.END, "DETAILED TRADE ANALYSIS\n")
        text_widget.insert(tk.END, "=" * 80 + "\n\n")

        # Process each trade
        total_profit = 0
        winning_trades = 0
        losing_trades = 0
        largest_win = 0
        largest_loss = 0
        
        for i, trade in enumerate(trades, 1):
            if trade['action'] == 'buy':
                text_widget.insert(tk.END, f"Trade #{i//2 + 1}\n")
                text_widget.insert(tk.END, "-" * 40 + "\n")
                text_widget.insert(tk.END, f"Entry Date: {trade['date'].strftime('%Y-%m-%d %H:%M')}\n")
                text_widget.insert(tk.END, f"Entry Price: ${trade['price']:.2f}\n")
                text_widget.insert(tk.END, f"Confidence: {trade['confidence']:.2%}\n")
            else:  # sell
                text_widget.insert(tk.END, f"Exit Date: {trade['date'].strftime('%Y-%m-%d %H:%M')}\n")
                text_widget.insert(tk.END, f"Exit Price: ${trade['price']:.2f}\n")
                text_widget.insert(tk.END, f"P/L: ${trade['pnl']:.2f}\n")
                text_widget.insert(tk.END, f"Return: {(trade['pnl']/trade['price']*100):.2f}%\n\n")
                
                # Update statistics
                total_profit += trade['pnl']
                if trade['pnl'] > 0:
                    winning_trades += 1
                    largest_win = max(largest_win, trade['pnl'])
                else:
                    losing_trades += 1
                    largest_loss = min(largest_loss, trade['pnl'])

        # Add summary statistics
        text_widget.insert(tk.END, "\nTRADE SUMMARY\n")
        text_widget.insert(tk.END, "=" * 80 + "\n\n")
        text_widget.insert(tk.END, f"Total Trades: {len(trades)//2}\n")
        text_widget.insert(tk.END, f"Winning Trades: {winning_trades}\n")
        text_widget.insert(tk.END, f"Losing Trades: {losing_trades}\n")
        if winning_trades + losing_trades > 0:
            win_rate = winning_trades / ((len(trades)//2)) * 100
            text_widget.insert(tk.END, f"Win Rate: {win_rate:.2f}%\n")
        text_widget.insert(tk.END, f"Total Profit/Loss: ${total_profit:.2f}\n")
        text_widget.insert(tk.END, f"Largest Win: ${largest_win:.2f}\n")
        text_widget.insert(tk.END, f"Largest Loss: ${largest_loss:.2f}\n")
        
        if winning_trades > 0 and losing_trades > 0:
            profit_factor = abs(largest_win/largest_loss) if largest_loss != 0 else float('inf')
            text_widget.insert(tk.END, f"Profit Factor: {profit_factor:.2f}\n")

        # Make text widget read-only
        text_widget.configure(state='disabled')

    def submit(self):
        asset_type = self.asset_type.get()
        symbol = self.symbol_entry.get().strip().upper()
        
        if not symbol:
            messagebox.showerror("Error", "Please enter a symbol")
            return
        
        # Get data using DataCollection
        data = DataCollection.get_asset_data(symbol, asset_type)
        
        if isinstance(data, str) and "Error" in data:
            messagebox.showerror("Error", data)
            return
            
        # Create new window for detailed information
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"{symbol} Analysis")
        detail_window.geometry("800x600")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(detail_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Market Data tab
        market_frame = ttk.Frame(notebook)
        notebook.add(market_frame, text='Market Data')
        
        # Display market data
        market_text = ""
        for key, value in data['market_data'].items():
            market_text += f"{key}: {value}\n"
        
        market_label = ttk.Label(market_frame, text=market_text, justify=tk.LEFT)
        market_label.pack(padx=10, pady=10)
        
        # Technical Analysis tab
        tech_frame = ttk.Frame(notebook)
        notebook.add(tech_frame, text='Technical Analysis')
        
        # Create price chart
        fig, ax = plt.subplots(figsize=(10, 6))
        hist_data = data['historical_data']
        
        # Plot price and moving averages
        ax.plot(hist_data.index, hist_data['Close'], label='Price')
        ax.plot(hist_data.index, hist_data['SMA_50'], label='50 SMA')
        ax.plot(hist_data.index, hist_data['SMA_200'], label='200 SMA')
        
        ax.set_title(f'{symbol} Price History')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        
        # Embed the plot in the GUI
        canvas = FigureCanvasTkAgg(fig, master=tech_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Trend Analysis tab
        trend_frame = ttk.Frame(notebook)
        notebook.add(trend_frame, text='Trend Analysis')
        
        if 'training_sets' in data:
            # Train the model
            training_results = self.trainer.train(data['training_sets'])
            
            if isinstance(training_results, dict):
                # Create a frame for model performance
                model_frame = ttk.LabelFrame(trend_frame, text="Model Performance")
                model_frame.pack(fill='x', padx=10, pady=5)
                
                # Display training metrics
                metrics_text = (
                    f"Training Accuracy: {training_results['train_accuracy']:.2%}\n"
                    f"Testing Accuracy: {training_results['test_accuracy']:.2%}\n"
                )
                ttk.Label(model_frame, text=metrics_text, justify=tk.LEFT).pack(padx=10, pady=5)
                
                # Make prediction for current trend
                latest_features = data['training_sets']['all']['features'][-1:]
                prediction = self.trainer.predict(latest_features)
                
                if isinstance(prediction, dict):
                    # Create prediction frame
                    pred_frame = ttk.LabelFrame(trend_frame, text="Current Prediction")
                    pred_frame.pack(fill='x', padx=10, pady=5)
                    
                    trend = "Positive" if prediction['prediction'] == 1 else "Negative"
                    pred_text = (
                        f"Trend Prediction: {trend}\n"
                        f"Confidence: {prediction['probability']:.2%}\n"
                    )
                    ttk.Label(pred_frame, text=pred_text, justify=tk.LEFT).pack(padx=10, pady=5)
                    
                    # Create feature analysis frame
                    feature_frame = ttk.LabelFrame(trend_frame, text="Feature Analysis")
                    feature_frame.pack(fill='x', padx=10, pady=5)
                    
                    # Create two columns: one for current values, one for importance
                    values_frame = ttk.Frame(feature_frame)
                    values_frame.pack(fill='x', padx=10, pady=5)
                    
                    # Left column: Current Values
                    current_frame = ttk.LabelFrame(values_frame, text="Current Values")
                    current_frame.pack(side='left', fill='x', expand=True, padx=5)
                    
                    for feature, value in prediction['feature_values'].items():
                        ttk.Label(
                            current_frame, 
                            text=f"{feature}: {value:.4f}",
                            justify=tk.LEFT
                        ).pack(anchor='w', padx=5)
                    
                    # Right column: Feature Importance
                    if hasattr(self.trainer, 'feature_importance'):
                        importance_frame = ttk.LabelFrame(values_frame, text="Feature Importance")
                        importance_frame.pack(side='left', fill='x', expand=True, padx=5)
                        
                        for feature, importance in self.trainer.feature_importance.items():
                            ttk.Label(
                                importance_frame,
                                text=f"{feature}: {importance:.4f}",
                                justify=tk.LEFT
                            ).pack(anchor='w', padx=5)
                else:
                    ttk.Label(trend_frame, text=f"Error: {prediction}").pack(padx=10, pady=10)
            else:
                ttk.Label(trend_frame, text=f"Error: {training_results}").pack(padx=10, pady=10)
        
        # Add Backtesting tab
        backtest_frame = ttk.Frame(notebook)
        notebook.add(backtest_frame, text='Backtesting')
        
        # Add backtest controls
        controls_frame = ttk.Frame(backtest_frame)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(controls_frame, text="Initial Capital:").pack(side='left', padx=5)
        capital_entry = ttk.Entry(controls_frame, width=10)
        capital_entry.insert(0, "10000")
        capital_entry.pack(side='left', padx=5)
        
        def run_backtest():
            try:
                initial_capital = float(capital_entry.get())
                self.backtest_results = self.trainer.backtest(data['historical_data'], initial_capital)
                
                if isinstance(self.backtest_results, dict):
                    # Clear previous results
                    for widget in results_frame.winfo_children():
                        widget.destroy()
                    
                    # Create summary frame
                    summary_frame = ttk.LabelFrame(results_frame, text="Performance Summary")
                    summary_frame.pack(fill='x', padx=10, pady=5)
                    
                    # Summary statistics
                    summary_text = (
                        f"Initial Capital: ${initial_capital:,.2f}\n"
                        f"Final Capital: ${self.backtest_results['final_capital']:,.2f}\n"
                        f"Total Return: {self.backtest_results['total_return']:.2f}%\n"
                        f"Number of Trades: {self.backtest_results['number_of_trades']}\n"
                        f"Winning Trades: {self.backtest_results['winning_trades']}\n"
                        f"Win Rate: {(self.backtest_results['winning_trades'] / self.backtest_results['number_of_trades'] * 100):.2f}%\n"
                        f"Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.2f}\n"
                        f"Max Drawdown: {self.backtest_results['max_drawdown']:.2f}%\n"
                        f"Average Trade Duration: {self.backtest_results.get('avg_trade_duration', 'N/A')}\n"
                        f"Profit Factor: {self.backtest_results.get('profit_factor', 'N/A')}"
                    )
                    
                    ttk.Label(summary_frame, text=summary_text, justify=tk.LEFT).pack(padx=10, pady=5)
                    
                    # Add button to view detailed trade history
                    ttk.Button(
                        results_frame, 
                        text="View Detailed Trade History",
                        command=lambda: self.create_trade_details_window(self.backtest_results['trades'])
                    ).pack(pady=5)
                    
                    # Plot equity curve
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])
                    
                    # Equity curve
                    equity_df = pd.DataFrame(self.backtest_results['equity_curve'])
                    ax1.plot(equity_df['date'], equity_df['equity'])
                    ax1.set_title('Equity Curve')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Portfolio Value ($)')
                    
                    # Drawdown plot
                    peak = equity_df['equity'].expanding(min_periods=1).max()
                    drawdown = (equity_df['equity'] - peak) / peak * 100
                    ax2.fill_between(equity_df['date'], drawdown, 0, color='red', alpha=0.3)
                    ax2.set_title('Drawdown (%)')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Drawdown %')
                    
                    plt.tight_layout()
                    
                    # Embed the plot
                    canvas = FigureCanvasTkAgg(fig, master=results_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
                    
                else:
                    messagebox.showerror("Error", str(self.backtest_results))
                    
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid initial capital amount")
        
        ttk.Button(controls_frame, text="Run Backtest", command=run_backtest).pack(side='left', padx=20)
        
        # Create frame for results
        results_frame = ttk.Frame(backtest_frame)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)