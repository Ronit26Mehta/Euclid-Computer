"""
Improved Frontend for the Financial Data Analytics Project

Updates include:
- A dedicated Investment Advisor tab that collects user risk tolerance and goals,
  then displays tailored advice along with additional context.
- An enhanced Anomaly Detection tab that shows computed z‑score, latest return,
  and flags anomalies.
- Continued integration with forecast, arbitrage, and portfolio optimization features.
- Improved UI/UX with custom CSS, responsive layouts, and error handling.
"""

import streamlit as st
import requests, json, time, logging
import pandas as pd
import plotly.graph_objs as go
from pathlib import Path
from datetime import datetime
from textblob import TextBlob
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_frontend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://127.0.0.1:5000/api"
CSV_FILES = [
    "cryptos.csv", "currencies.csv", "equities.csv",
    "etfs.csv", "funds.csv", "indices.csv", "moneymarkets.csv"
]
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Simple cache handler for API responses
class DataCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_duration = 3600  # seconds

    def get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def is_cache_valid(self, cache_path: Path) -> bool:
        return cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < self.cache_duration

    def get(self, key: str):
        cache_path = self.get_cache_path(key)
        if self.is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Cache read error: {str(e)}")
        return None

    def set(self, key: str, data):
        try:
            with open(self.get_cache_path(key), 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Cache write error: {str(e)}")

cache = DataCache(CACHE_DIR)

def check_api_health() -> bool:
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API health check error: {str(e)}")
        return False

@st.cache_data(ttl=3600)
def get_countries() -> list:
    try:
        response = requests.get(f'{API_BASE_URL}/countries', timeout=10)
        response.raise_for_status()
        return response.json()['countries']
    except Exception as e:
        logger.error(f"Countries fetch error: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_symbols(csv_file: str, country: str = None) -> list:
    try:
        cache_key = f"symbols_{csv_file}_{country}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        url = f"{API_BASE_URL}/csv/{csv_file}"
        if country and country != "All":
            url += f"?country={country}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        symbols = df.iloc[:, 0].dropna().unique().tolist()
        cache.set(cache_key, symbols)
        return symbols
    except Exception as e:
        logger.error(f"Symbols fetch error: {str(e)}")
        return []

def display_model_metrics(results: dict) -> pd.DataFrame:
    try:
        df = pd.DataFrame([
            {"Model": model,
             "RMSE": details.get('rmse', None),
             "MSE": details.get('mse', None),
             "MAE": details.get('mae', None),
             "R²": details.get('r2', None),
             "MAPE (%)": details.get('mape', None),
             "Status": "Success" if 'mse' in details else "Failed"}
            for model, details in results.items()
        ])
        return df.sort_values("RMSE")
    except Exception as e:
        logger.error(f"Metrics display error: {str(e)}")
        return pd.DataFrame()

def plot_model_comparison(results: dict) -> go.Figure:
    try:
        metrics = ['RMSE', 'MSE', 'MAE', 'MAPE']
        models = []
        metric_values = {metric: [] for metric in metrics}
        for model, details in results.items():
            if 'mse' in details:
                models.append(model)
                for metric in metrics:
                    metric_values[metric].append(details.get(metric.lower(), None))
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=metric_values[metric],
                text=[f'{v:.2f}' if v is not None else 'N/A' for v in metric_values[metric]],
                textposition='auto'
            ))
        fig.update_layout(title='Model Performance Comparison', barmode='group',
                          xaxis_title='Models', yaxis_title='Metric Value', showlegend=True)
        return fig
    except Exception as e:
        logger.error(f"Comparison plot error: {str(e)}")
        return go.Figure()

def analyze_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}

def fetch_company_info(symbol: str) -> dict:
    try:
        stock = yf.Ticker(symbol)
        return stock.info if stock.info else {}
    except Exception as e:
        logger.error(f"Company info error for {symbol}: {str(e)}")
        return {}

def main():
    st.set_page_config(page_title="Financial Prediction Dashboard", page_icon="📈",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        .custom-container {padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin-bottom: 1rem;}
        .plot-container {background-color: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 Advanced Financial Prediction Dashboard")
    
    if not check_api_health():
        st.error("API service is not available. Please try again later.")
        st.stop()
    
    # Sidebar configuration for data selection and model parameters
    with st.sidebar:
        st.header("Configuration")
        st.subheader("Data Selection")
        countries = get_countries()
        country_choice = st.selectbox("Select Country", ["All"] + countries,
                                      format_func=lambda x: "Global (No Country Specified)" if x=="All" else x)
        csv_choice = st.selectbox("Select Asset Type", CSV_FILES, format_func=lambda x: x.replace('.csv', '').title())
        st.subheader("Model Parameters")
        prediction_period = st.slider("Prediction Period (days)", 7, 365, 30)
        confidence_interval = st.slider("Confidence Interval (%)", 80, 99, 95)
        train_test_split = st.slider("Train-Test Split (%)", 50, 90, 80)
        seasonality_period = st.number_input("Seasonality Period (days)", 1, 365, 12)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Symbol Selection")
        with st.spinner("Loading symbols..."):
            selected_country = None if country_choice == "All" else country_choice
            symbols = get_symbols(csv_choice, selected_country)
        if not symbols:
            st.error("No symbols available for the selected criteria.")
            st.stop()
        symbol_choice = st.selectbox("Select Symbol", symbols)
        
        if st.button("Begin Prediction", key="predict", use_container_width=True):
            if symbol_choice:
                with st.spinner("Training models and generating predictions..."):
                    try:
                        response = requests.post(f"{API_BASE_URL}/train_model", json={
                            "symbol": symbol_choice,
                            "period": prediction_period,
                            "train_split": train_test_split / 100,
                            "seasonality": seasonality_period
                        }, timeout=120)
                        response.raise_for_status()
                        result = response.json()
                        if result["status"] == "success":
                            st.success("Model training completed successfully!")
                            st.session_state.prediction_results = result
                            st.session_state.selected_symbol = symbol_choice
                        else:
                            st.error(f"Training failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    # Expanded tabs for multiple functionalities
    with col2:
        tabs = st.tabs(["Forecast & Models", "Arbitrage", "Portfolio Optimizer", "Investment Advisor", "Anomaly Detection", "Company Info", "News & Sentiment"])
        
        # Tab 1: Forecast and Model Comparison
        with tabs[0]:
            if 'prediction_results' in st.session_state:
                result = st.session_state.prediction_results
                symbol = st.session_state.selected_symbol
                st.subheader("Model Comparison")
                metrics_df = display_model_metrics(result["results"])
                st.dataframe(metrics_df, use_container_width=True)
                if result.get("best_model"):
                    st.success(f"Best model: {result['best_model']}")
                st.plotly_chart(plot_model_comparison(result["results"]), use_container_width=True)
                st.subheader("Forecast Visualization")
                if "plot_url" in result:
                    try:
                        html_content = requests.get(f"http://127.0.0.1:5000{result['plot_url']}").text
                        st.components.v1.html(html_content, height=600)
                    except Exception as e:
                        st.error(f"Forecast visualization error: {str(e)}")
                st.subheader("Training Summary")
                summary = result.get("training_summary", {})
                cols = st.columns(3)
                cols[0].metric("Total Data Points", summary.get("data_points", 0))
                cols[1].metric("Training Size", summary.get("train_size", 0))
                cols[2].metric("Test Size", summary.get("test_size", 0))
                st.metric("Seasonal Strength", f"{result.get('seasonal_strength',0)*100:.2f}%")
                st.download_button("Download Results CSV", pd.DataFrame(result["results"]).to_csv(index=False),
                                   file_name=f"{symbol}_results.csv", mime="text/csv")
        
        # Tab 2: Arbitrage Detection
        with tabs[1]:
            st.subheader("Multi-Market Arbitrage Detection")
            if st.button("Run Arbitrage Detection", key="arbitrage"):
                try:
                    resp = requests.post(f"{API_BASE_URL}/arbitrage", json={"symbol": st.session_state.get("selected_symbol")}, timeout=30)
                    data = resp.json()
                    st.write("Arbitrage Data:", data)
                except Exception as e:
                    st.error(f"Arbitrage error: {str(e)}")
        
        # Tab 3: Portfolio Optimizer
        with tabs[2]:
            st.subheader("Smart Portfolio Optimizer")
            st.write("Enter your portfolio as a list of (symbol, weight) pairs (in JSON format).")
            portfolio_input = st.text_area("Portfolio", value='[["AAPL", 0.5], ["MSFT", 0.5]]')
            if st.button("Optimize Portfolio", key="optimize"):
                try:
                    portfolio = json.loads(portfolio_input)
                    resp = requests.post(f"{API_BASE_URL}/optimize_portfolio", json={"portfolio": portfolio}, timeout=30)
                    st.write("Optimized Portfolio:", resp.json().get("optimized_portfolio"))
                except Exception as e:
                    st.error(f"Portfolio optimization error: {str(e)}")
        
        # Tab 4: Personalized Investment Advisor
        with tabs[3]:
            st.subheader("Personalized Investment Advisor")
            st.write("Enter your investment profile:")
            risk = st.selectbox("Risk Tolerance", ["low", "medium", "high"])
            goals = st.text_input("Investment Goals", "Balanced growth with moderate risk")
            if st.button("Get Investment Advice", key="advisor"):
                try:
                    resp = requests.post(f"{API_BASE_URL}/personal_advice", json={"risk": risk, "goals": goals}, timeout=30)
                    advice = resp.json().get("advice")
                    if advice:
                        st.success("Investment Advice:")
                        st.info(advice)
                    else:
                        st.error("No advice received.")
                except Exception as e:
                    st.error(f"Investment advisor error: {str(e)}")
        
        # Tab 5: Real-Time Market Anomaly Detection
        with tabs[4]:
            st.subheader("Market Anomaly Detection")
            if st.button("Detect Anomalies", key="anomaly"):
                try:
                    resp = requests.post(f"{API_BASE_URL}/anomaly_detection", json={"symbol": st.session_state.get("selected_symbol")}, timeout=30)
                    anomaly_data = resp.json()
                    if "error" not in anomaly_data:
                        st.write("Anomaly Detection Results:")
                        st.json(anomaly_data)
                        # Highlight anomaly if detected
                        if anomaly_data.get("anomaly_detected"):
                            st.error("Anomaly detected! Check the z‑score and recent price movements.")
                        else:
                            st.success("No significant anomaly detected.")
                    else:
                        st.error(f"Error: {anomaly_data.get('error')}")
                except Exception as e:
                    st.error(f"Anomaly detection error: {str(e)}")
        
        # Tab 6: Company Info
        with tabs[5]:
            st.subheader("Company Information")
            if st.button("Fetch Company Info", key="company_info"):
                with st.spinner("Retrieving company info..."):
                    company_data = fetch_company_info(st.session_state.get("selected_symbol"))
                    if not company_data:
                        st.warning("No company info available.")
                    else:
                        info_df = pd.DataFrame([{"Field": k, "Value": v} for k, v in company_data.items()])
                        st.table(info_df)
        
        # Tab 7: News & Sentiment
        with tabs[6]:
            st.subheader("News & Sentiment")
            if st.button("Fetch Latest News", key="news"):
                with st.spinner("Fetching news articles..."):
                    try:
                        stock = yf.Ticker(st.session_state.get("selected_symbol"))
                        news_articles = stock.news
                        if not news_articles:
                            st.warning("No news available.")
                        else:
                            news_list = []
                            for article in news_articles:
                                title = article.get("title", "N/A")
                                link = article.get("link", "#")
                                sentiment = analyze_sentiment(title)
                                news_list.append({
                                    "Title": title,
                                    "Publisher": article.get("publisher", "Unknown"),
                                    "Published": article.get("providerPublishTime", "N/A"),
                                    "Sentiment": f"Polarity: {sentiment['polarity']:.2f}, Subjectivity: {sentiment['subjectivity']:.2f}",
                                    "Link": link
                                })
                            news_df = pd.DataFrame(news_list)
                            st.dataframe(news_df.drop(columns=["Link"]), use_container_width=True)
                            st.markdown("### News Links")
                            for _, row in news_df.iterrows():
                                st.markdown(f"- [{row['Title']}]({row['Link']}) by {row['Publisher']}")
                    except Exception as e:
                        st.error(f"News fetch error: {str(e)}")
    
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center'>
            <p>Made with ❤ | Data provided by Yahoo Finance | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

if __name__ == "__main__":
    main()
