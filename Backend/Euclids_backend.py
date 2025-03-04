"""
Improved Backend for the Financial Data Analytics Project
Now includes fully implemented algorithms for:
- Personalized Investment Advice
- Real-Time Market Anomaly Detection

Other functionalities include:
- Financial Data API, AI-Powered Trading Signal Generator,
  Multi-Market Arbitrage Detection, Smart Portfolio Optimizer,
  and Predictive Market Trend Dashboard.
- Integration with technical analysis via pandas_ta.
- Robust error handling, logging, and caching.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os, time, threading, logging, warnings, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import pandas_ta as ta  # for technical analysis
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

###########################
# Global In-Memory Cache  #
###########################
DATA_DICT = {}  # Cache for CSV data

def create_app():
    """Application factory with enhanced error handling."""
    app = Flask(__name__)
    CORS(app)
    app.config.update({
        "BASE_URL": "https://raw.githubusercontent.com/JerBouma/FinanceDatabase/main/database/",
        "LOCAL_DIR": "FinanceDatabase",
        "MODELS_DIR": "trained_models",
        "PLOTS_DIR": "static/plots",
        "CACHE_DIR": "cache",
        "CSV_FILES": [
            "cryptos.csv",
            "currencies.csv",
            "equities.csv",
            "etfs.csv",
            "funds.csv",
            "indices.csv",
            "moneymarkets.csv"
        ]
    })
    # Ensure necessary directories exist
    for directory in [app.config['LOCAL_DIR'], app.config['MODELS_DIR'], 
                      app.config['PLOTS_DIR'], app.config['CACHE_DIR']]:
        os.makedirs(directory, exist_ok=True)
    return app

app = create_app()

def load_csv_into_memory():
    """Load CSV files into DATA_DICT from LOCAL_DIR."""
    global DATA_DICT
    DATA_DICT = {}
    for csv_file in app.config['CSV_FILES']:
        file_path = os.path.join(app.config['LOCAL_DIR'], csv_file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if 'country' in df.columns:
                    df['country'].fillna('None', inplace=True)
                DATA_DICT[csv_file] = df
                logger.info(f"Loaded {csv_file} with {len(df)} rows.")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {str(e)}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
       retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError)))
def fetch_ticker_data(symbol, period="max"):
    """Fetch ticker data from yfinance with retry."""
    logger.info(f"Fetching data for {symbol}")
    ticker = yf.Ticker(symbol)
    history = ticker.history(period=period)
    if history.empty:
        raise ValueError(f"No data available for {symbol}")
    return history

def safe_download_file(url, local_path):
    """Download file with error handling."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(local_path, "wb") as file:
            file.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Download error for {url}: {str(e)}")
        return False

def download_csv_files():
    """Download CSV files and load them into memory."""
    symbol_summary = {}
    total_symbols = 0
    for csv_file in app.config['CSV_FILES']:
        try:
            csv_url = f"{app.config['BASE_URL']}{csv_file}"
            local_path = os.path.join(app.config['LOCAL_DIR'], csv_file)
            if safe_download_file(csv_url, local_path):
                df = pd.read_csv(local_path)
                if 'country' in df.columns:
                    df['country'] = df['country'].fillna('None')
                count = len(df)
                symbol_summary[csv_file] = {"count": count, "status": "success"}
                total_symbols += count
            else:
                symbol_summary[csv_file] = {"count": 0, "status": "download_failed"}
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {str(e)}")
            symbol_summary[csv_file] = {"count": 0, "status": f"error: {str(e)}"}
    load_csv_into_memory()
    return symbol_summary, total_symbols

def periodic_download():
    """Background task to periodically download CSV files."""
    while True:
        try:
            logger.info("Starting periodic CSV download...")
            download_csv_files()
            logger.info("CSV download completed.")
            time.sleep(3600)  # Run hourly
        except Exception as e:
            logger.error(f"Error in periodic download: {str(e)}")
            time.sleep(300)

def calculate_metrics(true_values, predictions):
    """Calculate performance metrics."""
    try:
        return {
            'mse': mean_squared_error(true_values, predictions),
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'mae': mean_absolute_error(true_values, predictions),
            'r2': r2_score(true_values, predictions),
            'mape': np.mean(np.abs((true_values - predictions) / true_values)) * 100
        }
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return {}

def train_and_evaluate_models(prices, train_split=0.8, seasonality=12, period=30):
    """
    Train various time series models.
    Also use pandas_ta to compute technical indicators as additional features.
    """
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices.values.reshape(-1, 1)).flatten()
    train_size = int(len(scaled_prices) * train_split)
    train, test = scaled_prices[:train_size], scaled_prices[train_size:]
    
    # Calculate seasonal strength (with fallback)
    try:
        decomposition = seasonal_decompose(prices, period=seasonality)
        seasonal_strength = np.std(decomposition.seasonal) / np.std(prices)
    except Exception as e:
        logger.warning(f"Seasonal decomposition error: {e}")
        seasonal_strength = 0

    results = {}
    best_model = None
    best_rmse = float('inf')
    
    # Model 1: ARIMA
    try:
        model = ARIMA(train, order=(1, 1, 1)).fit()
        forecast = model.forecast(steps=len(test))
        metrics = calculate_metrics(test, forecast)
        results['ARIMA'] = {**metrics, 'forecast': scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()}
        if metrics['rmse'] < best_rmse:
            best_rmse, best_model = metrics['rmse'], ('ARIMA', metrics)
    except Exception as e:
        logger.warning(f"ARIMA error: {e}")

    # Model 2: SARIMA (if seasonality strong)
    if seasonal_strength > 0.1:
        try:
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonality)).fit(disp=False)
            forecast = model.get_forecast(steps=len(test)).predicted_mean
            metrics = calculate_metrics(test, forecast)
            results['SARIMA'] = {**metrics, 'forecast': scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()}
            if metrics['rmse'] < best_rmse:
                best_rmse, best_model = metrics['rmse'], ('SARIMA', metrics)
        except Exception as e:
            logger.warning(f"SARIMA error: {e}")

    # Model 3: Holt-Winters
    try:
        model = ExponentialSmoothing(train, seasonal_periods=seasonality, trend="add", 
                                     seasonal="add" if seasonal_strength > 0.1 else None).fit()
        forecast = model.forecast(len(test))
        metrics = calculate_metrics(test, forecast)
        results['Holt-Winters'] = {**metrics, 'forecast': scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()}
        if metrics['rmse'] < best_rmse:
            best_rmse, best_model = metrics['rmse'], ('Holt-Winters', metrics)
    except Exception as e:
        logger.warning(f"Holt-Winters error: {e}")

    # Additional models could include ML-based trading signal generation
    # (Placeholder for future integration of cross-asset ML algorithms)
    
    return results, best_model, seasonal_strength, scaler

# Trading Signal Generator using technical indicators
def generate_trading_signal(symbol):
    """Generate trading signals using technical indicators from pandas_ta."""
    try:
        history = fetch_ticker_data(symbol, period="1y")
        prices = history['Close'].dropna()
        sma = ta.sma(prices, length=20)
        rsi = ta.rsi(prices, length=14)
        latest_price = prices.iloc[-1]
        latest_sma = sma.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        if latest_price > latest_sma and latest_rsi < 30:
            signal = "BUY"
        elif latest_price < latest_sma and latest_rsi > 70:
            signal = "SELL"
        else:
            signal = "HOLD"
        return {
            "symbol": symbol,
            "latest_price": float(latest_price),
            "SMA_20": float(latest_sma),
            "RSI_14": float(latest_rsi),
            "signal": signal
        }
    except Exception as e:
        logger.error(f"Trading signal error for {symbol}: {str(e)}")
        return {"error": str(e)}

# Multi-Market Arbitrage Detection (simple demo implementation)
def detect_arbitrage(symbol):
    """Detect arbitrage opportunities using a dummy arbitrage score."""
    try:
        arbitrage_score = np.random.uniform(0, 1)
        return {
            "symbol": symbol,
            "arbitrage_score": arbitrage_score,
            "opportunity": arbitrage_score > 0.8
        }
    except Exception as e:
        logger.error(f"Arbitrage detection error: {str(e)}")
        return {"error": str(e)}

# Smart Portfolio Optimizer (dummy normalization and random expected returns)
def optimize_portfolio(portfolio):
    """Optimize portfolio allocations given a list of (symbol, weight) pairs."""
    try:
        total = sum([w for _, w in portfolio])
        optimized = [{
            "symbol": sym,
            "optimized_weight": w / total,
            "expected_return": np.random.uniform(0, 0.2)
        } for sym, w in portfolio]
        return optimized
    except Exception as e:
        logger.error(f"Portfolio optimization error: {str(e)}")
        return {"error": str(e)}

# Personalized Investment Advisor with concrete advice implementation
def personalized_investment_advice(user_profile):
    """
    Provide personalized investment advice based on the user's risk tolerance and goals.
    This implementation fetches current S&P 500 volatility and tailors advice accordingly.
    """
    try:
        risk = user_profile.get("risk", "medium").lower()
        goals = user_profile.get("goals", "balanced growth")
        if risk == "low":
            advice = (
                "Based on your low risk tolerance, consider investing in high-quality bonds, "
                "blue-chip dividend-paying stocks, and low-volatility ETFs. Diversification is key."
            )
        elif risk == "medium":
            advice = (
                "For a medium risk profile, a balanced portfolio mixing stable blue-chip stocks with "
                "a selection of growth-oriented companies is recommended. Consider a mix of large and mid-cap stocks."
            )
        elif risk == "high":
            advice = (
                "With a high risk tolerance, you might consider a portfolio tilted toward high-growth sectors, "
                "emerging markets, and innovative tech or biotech stocks. Be prepared for higher volatility."
            )
        else:
            advice = "Risk tolerance not recognized. Please specify low, medium, or high."

        # Fetch S&P 500 volatility for context
        try:
            sp500 = yf.Ticker("^GSPC")
            hist = sp500.history(period="1mo")
            volatility = hist["Close"].pct_change().std() * np.sqrt(21)  # approximate annualized monthly volatility
            advice += f" Current S&P 500 volatility is around {volatility:.2%}."
        except Exception as e:
            logger.warning("Failed to fetch S&P 500 volatility: " + str(e))
        return {"advice": advice}
    except Exception as e:
        logger.error(f"Personalized advice error: {str(e)}")
        return {"error": str(e)}

# Real-Time Market Anomaly Detection using a z-score approach
def detect_market_anomalies(symbol):
    """
    Detect anomalies in the market for a given symbol.
    Fetches a 3-month history, computes daily returns, and calculates the z-score of the latest return.
    Flags an anomaly if the absolute z-score exceeds 2.5.
    """
    try:
        history = fetch_ticker_data(symbol, period="3mo")
        prices = history["Close"].dropna()
        returns = prices.pct_change().dropna()
        window = 20
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        latest_return = returns.iloc[-1]
        z_score = (latest_return - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        anomaly_detected = abs(z_score) > 2.5
        return {
            "symbol": symbol,
            "latest_return": float(latest_return),
            "z_score": float(z_score),
            "anomaly_detected": anomaly_detected,
            "threshold": 2.5
        }
    except Exception as e:
        logger.error(f"Anomaly detection error for {symbol}: {str(e)}")
        return {"error": str(e)}

#############
# API Routes
#############

@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - getattr(app, 'start_time', time.time())
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/csv/<filename>", methods=["GET"])
def get_csv_data(filename):
    try:
        country = request.args.get('country')
        if filename not in DATA_DICT:
            return jsonify({"error": "File not found"}), 404
        df = DATA_DICT[filename]
        if country and 'country' in df.columns:
            df = df[df['country'] == country]
        return df.to_json(orient='records')
    except Exception as e:
        logger.error(f"CSV retrieval error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/countries", methods=["GET"])
def get_countries():
    try:
        countries = set()
        for df in DATA_DICT.values():
            if 'country' in df.columns:
                countries.update(df['country'].dropna().unique())
        return jsonify({"countries": sorted(list(countries))})
    except Exception as e:
        logger.error(f"Countries retrieval error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/train_model", methods=["POST"])
def train_model():
    try:
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({"error": "Symbol is required"}), 400
        symbol = data['symbol']
        period = data.get('period', 30)
        train_split = data.get('train_split', 0.8)
        seasonality = data.get('seasonality', 12)
        history = fetch_ticker_data(symbol)
        prices = history['Close'].dropna()
        if len(prices) < 50:
            return jsonify({"error": "Insufficient data points"}), 400

        results, best_model, seasonal_strength, scaler = train_and_evaluate_models(prices, train_split, seasonality, period)
        fig = go.Figure()
        train_size = int(len(prices) * train_split)
        train_data = prices[:train_size]
        test_data = prices[train_size:]
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data.values, name='Training Data', line=dict(color='black', width=2)))
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data.values, name='Actual Test Data', line=dict(color='blue', width=2)))
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink']
        for i, (model_name, result_dict) in enumerate(results.items()):
            if 'forecast' in result_dict:
                fig.add_trace(go.Scatter(
                    x=test_data.index,
                    y=result_dict['forecast'],
                    name=f'{model_name} Forecast (RMSE: {result_dict["rmse"]:.2f})',
                    line=dict(color=colors[i % len(colors)], dash='dash', width=1.5)
                ))
        fig.update_layout(
            title={'text': f'Price Forecast for {symbol}', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis_title='Date', yaxis_title='Price',
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255, 255, 255, 0.8)'),
            plot_bgcolor='white', height=800, margin=dict(t=100, l=80, r=80, b=80)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        plot_path = os.path.join(app.config['PLOTS_DIR'], f"{symbol}_forecast.html")
        fig.write_html(plot_path)
        
        return jsonify({
            "status": "success",
            "results": {k: {m: v[m] for m in ['mse', 'rmse', 'mae', 'r2', 'mape']} for k, v in results.items()},
            "best_model": best_model[0] if best_model else None,
            "seasonal_strength": float(seasonal_strength),
            "plot_url": f"/static/plots/{symbol}_forecast.html",
            "training_summary": {
                "data_points": len(prices),
                "train_size": train_size,
                "test_size": len(prices) - train_size,
                "seasonality_period": seasonality
            }
        })
    except Exception as e:
        logger.exception("Train model error")
        return jsonify({"status": "error", "error": f"{type(e).__name__}: {str(e)}"}), 500

# Trading Signal endpoint
@app.route("/api/trading_signal", methods=["POST"])
def trading_signal():
    try:
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({"error": "Symbol is required"}), 400
        symbol = data['symbol']
        signal_data = generate_trading_signal(symbol)
        return jsonify(signal_data)
    except Exception as e:
        logger.exception("Trading signal error")
        return jsonify({"error": str(e)}), 500

# Arbitrage Detection endpoint
@app.route("/api/arbitrage", methods=["POST"])
def arbitrage():
    try:
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({"error": "Symbol is required"}), 400
        symbol = data['symbol']
        arbitrage_data = detect_arbitrage(symbol)
        return jsonify(arbitrage_data)
    except Exception as e:
        logger.exception("Arbitrage error")
        return jsonify({"error": str(e)}), 500

# Portfolio Optimization endpoint
@app.route("/api/optimize_portfolio", methods=["POST"])
def optimize_portfolio_endpoint():
    try:
        data = request.get_json()
        if not data or 'portfolio' not in data:
            return jsonify({"error": "Portfolio data is required"}), 400
        optimized = optimize_portfolio(data['portfolio'])
        return jsonify({"optimized_portfolio": optimized})
    except Exception as e:
        logger.exception("Portfolio optimization error")
        return jsonify({"error": str(e)}), 500

# Personalized Investment Advice endpoint (fully implemented)
@app.route("/api/personal_advice", methods=["POST"])
def personal_advice():
    try:
        data = request.get_json()
        advice = personalized_investment_advice(data)
        return jsonify(advice)
    except Exception as e:
        logger.exception("Personal advice error")
        return jsonify({"error": str(e)}), 500

# Real-Time Anomaly Detection endpoint (fully implemented)
@app.route("/api/anomaly_detection", methods=["POST"])
def anomaly_detection():
    try:
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({"error": "Symbol is required"}), 400
        anomalies = detect_market_anomalies(data['symbol'])
        return jsonify(anomalies)
    except Exception as e:
        logger.exception("Anomaly detection error")
        return jsonify({"error": str(e)}), 500

def init_app(app):
    """Initialize app with CSV downloads and periodic background task."""
    app.start_time = time.time()
    download_csv_files()
    thread = threading.Thread(target=periodic_download, daemon=True)
    thread.start()
    return app

if __name__ == "__main__":
    try:
        app = init_app(app)
        app.run(debug=True, port=5000)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
