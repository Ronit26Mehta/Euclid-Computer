from flask import Flask, jsonify, request
import os
import requests
import pandas as pd
import yfinance as yf
import threading
import time
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from flask_cors import CORS
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
# Time Series Models from Statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

###########################
# Global In-Memory Cache  #
###########################
DATA_DICT = {}  # { "cryptos.csv": pd.DataFrame, "equities.csv": pd.DataFrame, ... }

def create_app():
    """Application factory function with enhanced error handling"""
    try:
        app = Flask(__name__)
        CORS(app)
        
        # Configuration
        app.config.update(
            BASE_URL="https://raw.githubusercontent.com/JerBouma/FinanceDatabase/main/database/",
            LOCAL_DIR="FinanceDatabase",
            MODELS_DIR="trained_models",
            PLOTS_DIR="static/plots",
            CACHE_DIR="cache",
            CSV_FILES=[
                "cryptos.csv",
                "currencies.csv",
                "equities.csv",
                "etfs.csv",
                "funds.csv",
                "indices.csv",
                "moneymarkets.csv"
            ]
        )
        
        # Create necessary directories
        for directory in [
            app.config['LOCAL_DIR'],
            app.config['MODELS_DIR'],
            app.config['PLOTS_DIR'],
            app.config['CACHE_DIR']
        ]:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {str(e)}")
        
        return app
    except Exception as e:
        logger.error(f"Error in create_app: {str(e)}")
        raise

app = create_app()

def load_csv_into_memory():
    """
    Reads CSV files from disk (LOCAL_DIR) into global DATA_DICT once.
    This allows the server to respond quickly to symbol/country requests.
    """
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
                logger.info(f"Loaded {csv_file} into memory with {len(df)} rows.")
            except Exception as e:
                logger.error(f"Error loading {csv_file} into memory: {str(e)}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError))
)
def fetch_ticker_data(symbol, period="max"):
    """Fetch ticker data with retry logic."""
    try:
        logger.info(f"Fetching data for symbol: {symbol}")
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period)
        
        if history.empty:
            raise ValueError(f"No data available for symbol {symbol}")
            
        return history
    except Exception as e:
        logger.error(f"Error fetching ticker data for {symbol}: {str(e)}")
        raise

def safe_download_file(url, local_path):
    """Safely download a file with error handling."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(local_path, "wb") as file:
            file.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def download_csv_files():
    """Download CSV files with improved error handling, then load into memory."""
    try:
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
                    symbol_count = len(df)
                    symbol_summary[csv_file] = {
                        "count": symbol_count,
                        "status": "success"
                    }
                    total_symbols += symbol_count
                else:
                    symbol_summary[csv_file] = {
                        "count": 0,
                        "status": "download_failed"
                    }
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {str(e)}")
                symbol_summary[csv_file] = {
                    "count": 0,
                    "status": f"error: {str(e)}"
                }

        # After downloading, load all CSVs into memory
        load_csv_into_memory()

        return symbol_summary, total_symbols
    except Exception as e:
        logger.error(f"Error in download_csv_files: {str(e)}")
        return {}, 0

def periodic_download():
    """Background task for periodic downloads."""
    while True:
        try:
            logger.info("Starting periodic download...")
            download_csv_files()
            logger.info("Periodic download completed.")
            time.sleep(3600)  # Wait for 1 hour
        except Exception as e:
            logger.error(f"Error in periodic download: {str(e)}")
            time.sleep(300)  # Wait for 5 minutes before retry

def calculate_metrics(true_values, predictions):
    """Calculate various performance metrics."""
    try:
        return {
            'mse': mean_squared_error(true_values, predictions),
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'mae': mean_absolute_error(true_values, predictions),
            'r2': r2_score(true_values, predictions),
            'mape': np.mean(np.abs((true_values - predictions) / true_values)) * 100
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {}

def train_and_evaluate_models(prices, train_split=0.8, seasonality=12, period=30):
    """Train and evaluate multiple time series models using Statsmodels."""
    try:
        # Scale the data
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices.values.reshape(-1, 1)).flatten()

        # Train-test split
        train_size = int(len(scaled_prices) * train_split)
        train = scaled_prices[:train_size]
        test = scaled_prices[train_size:]

        # Calculate seasonal strength
        try:
            decomposition = seasonal_decompose(prices, period=seasonality)
            seasonal_strength = np.std(decomposition.seasonal) / np.std(prices)
        except Exception as e:
            logger.warning(f"Seasonal decomposition failed: {e}")
            seasonal_strength = 0

        results = {}
        best_model = None
        best_rmse = float('inf')

        ### Models ###

        # 1. ARIMA
        try:
            arima_model = ARIMA(train, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=len(test))
            metrics = calculate_metrics(test, arima_forecast)
            results['ARIMA'] = {
                **metrics,
                'forecast': scaler.inverse_transform(arima_forecast.reshape(-1, 1)).flatten()
            }
            if metrics['rmse'] < best_rmse:
                best_rmse, best_model = metrics['rmse'], ('ARIMA', metrics)
        except Exception as e:
            logger.warning(f"ARIMA failed: {e}")

        # 2. SARIMA (with seasonality)
        if seasonal_strength > 0.1:
            try:
                sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonality))
                sarima_fit = sarima_model.fit(disp=False)
                sarima_forecast = sarima_fit.get_forecast(steps=len(test)).predicted_mean
                metrics = calculate_metrics(test, sarima_forecast)
                results['SARIMA'] = {
                    **metrics,
                    'forecast': scaler.inverse_transform(sarima_forecast.reshape(-1, 1)).flatten()
                }
                if metrics['rmse'] < best_rmse:
                    best_rmse, best_model = metrics['rmse'], ('SARIMA', metrics)
            except Exception as e:
                logger.warning(f"SARIMA failed: {e}")

        # 3. Holt-Winters
        try:
            hw_model = ExponentialSmoothing(
                train,
                seasonal_periods=seasonality,
                trend="add",
                seasonal="add" if seasonal_strength > 0.1 else None
            )
            hw_fit = hw_model.fit()
            hw_forecast = hw_fit.forecast(len(test))
            metrics = calculate_metrics(test, hw_forecast)
            results['Holt-Winters'] = {
                **metrics,
                'forecast': scaler.inverse_transform(hw_forecast.reshape(-1, 1)).flatten()
            }
            if metrics['rmse'] < best_rmse:
                best_rmse, best_model = metrics['rmse'], ('Holt-Winters', metrics)
        except Exception as e:
            logger.warning(f"Holt-Winters failed: {e}")

        # 4. Simple Exponential Smoothing
        try:
            ses_model = SimpleExpSmoothing(train).fit()
            ses_forecast = ses_model.forecast(len(test))
            metrics = calculate_metrics(test, ses_forecast)
            results['Simple Exponential Smoothing'] = {
                **metrics,
                'forecast': scaler.inverse_transform(ses_forecast.reshape(-1, 1)).flatten()
            }
            if metrics['rmse'] < best_rmse:
                best_rmse, best_model = metrics['rmse'], ('Simple Exponential Smoothing', metrics)
        except Exception as e:
            logger.warning(f"Simple Exponential Smoothing failed: {e}")

        # 5. Holt's Linear Trend
        try:
            holt_model = Holt(train).fit()
            holt_forecast = holt_model.forecast(len(test))
            metrics = calculate_metrics(test, holt_forecast)
            results['Holt Linear'] = {
                **metrics,
                'forecast': scaler.inverse_transform(holt_forecast.reshape(-1, 1)).flatten()
            }
            if metrics['rmse'] < best_rmse:
                best_rmse, best_model = metrics['rmse'], ('Holt Linear', metrics)
        except Exception as e:
            logger.warning(f"Holt's Linear Trend failed: {e}")

        return results, best_model, seasonal_strength, scaler
    except Exception as e:
        logger.error(f"Error in train_and_evaluate_models: {e}")
        raise


@app.route("/api/health", methods=["GET"])
def health_check():
    """API endpoint for health checking."""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - getattr(app, 'start_time', 0)
        })
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/api/csv/<filename>", methods=["GET"])
def get_csv_data(filename):
    """
    API endpoint to get CSV data with country filtering, served from in-memory cache (DATA_DICT).
    This is much faster than reading from disk every time.
    """
    try:
        country = request.args.get('country')
        # Use the in-memory dictionary to retrieve the DataFrame
        if filename not in DATA_DICT:
            return jsonify({"error": "File not found in memory"}), 404

        df = DATA_DICT[filename]

        if country and 'country' in df.columns:
            df = df[df['country'] == country]
            
        # Return as JSON for easier parsing by the frontend
        return df.to_json(orient='records')
    except Exception as e:
        logger.error(f"Error getting CSV data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/countries", methods=["GET"])
def get_countries():
    """
    API endpoint to get list of available countries across all CSVs.
    Uses the in-memory DATA_DICT for speed.
    """
    try:
        countries = set()
        for csv_file, df in DATA_DICT.items():
            if 'country' in df.columns:
                countries.update(df['country'].dropna().unique())
                
        return jsonify({"countries": sorted(list(countries))})
    except Exception as e:
        logger.error(f"Error getting countries: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/train_model", methods=["POST"])
def train_model():
    """Train and evaluate multiple statistical models."""
    try:
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({"error": "Symbol is required"}), 400

        symbol = data['symbol']
        period = data.get('period', 30)
        train_split = data.get('train_split', 0.8)
        seasonality = data.get('seasonality', 12)
        
        # Fetch data
        history = fetch_ticker_data(symbol)
        prices = history['Close'].dropna()
        
        if len(prices) < 50:
            return jsonify({"error": "Insufficient data points"}), 400

        # Train models
        results, best_model, seasonal_strength, scaler = train_and_evaluate_models(
            prices, train_split, seasonality, period
        )

        # Create visualization
        fig = go.Figure()
        
        # Calculate split point for visualization
        train_size = int(len(prices) * train_split)
        train_data = prices[:train_size]
        test_data = prices[train_size:]
        
        # Plot training data
        fig.add_trace(go.Scatter(
            x=train_data.index,
            y=train_data.values,
            name='Training Data',
            line=dict(color='black', width=2)
        ))
        
        # Plot test data
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data.values,
            name='Actual Test Data',
            line=dict(color='blue', width=2)
        ))

        # Plot model forecasts
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink']
        for i, (model_name, result_dict) in enumerate(results.items()):
            if 'forecast' in result_dict:
                fig.add_trace(go.Scatter(
                    x=test_data.index,
                    y=result_dict['forecast'],
                    name=f'{model_name} Forecast (RMSE: {result_dict["rmse"]:.2f})',
                    line=dict(
                        color=colors[i % len(colors)],
                        dash='dash',
                        width=1.5
                    )
                ))

        # Update layout with better formatting
        fig.update_layout(
            title={
                'text': f'Price Forecast for {symbol}',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            plot_bgcolor='white',
            height=800,
            margin=dict(t=100, l=80, r=80, b=80)
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

        # Save plot
        plot_path = os.path.join(app.config['PLOTS_DIR'], f"{symbol}_forecast.html")
        fig.write_html(plot_path)
        
        return jsonify({
            "status": "success",
            "results": {k: {metric: v[metric] for metric in ['mse', 'rmse', 'mae', 'r2', 'mape']} 
                       for k, v in results.items()},
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
        logger.exception("An error occurred in train_model")
        return jsonify({
            "status": "error",
            "error": f"{type(e)._name_}: {str(e)}"
        }), 500

def init_app(app):
    """Initialize the Flask application with enhanced error handling."""
    try:
        app.start_time = time.time()
        
        # 1. Ensure CSVs exist (download if missing) and load them into memory
        download_csv_files()
        
        # 2. Start background tasks for periodic re-download
        thread = threading.Thread(target=periodic_download, daemon=True)
        thread.start()
        
        return app
    except Exception as e:
        logger.error(f"Error initializing app: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Initialize the application
        app = init_app(app)
        
        # Run the Flask app
        app.run(debug=True, port=5000)
    except Exception as e:
        logger.error(f"Fatal error starting app: {str(e)}")