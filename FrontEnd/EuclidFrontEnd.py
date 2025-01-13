import streamlit as st
import requests
import pandas as pd
import plotly.graph_objs as go
from pathlib import Path
import time
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.express as px
import logging
from typing import Dict, List, Optional, Union

##############################
# NEW: We use yfinance.news
##############################
import yfinance as yf
from textblob import TextBlob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frontend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://127.0.0.1:5000/api"
CSV_FILES = [
    "cryptos.csv",
    "currencies.csv",
    "equities.csv",
    "etfs.csv",
    "funds.csv",
    "indices.csv",
    "moneymarkets.csv"
]

# Cache directory for temporary data (frontend cache)
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

class DataCache:
    """Handle caching of API responses and data on the frontend side."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_duration = 3600  # 1 hour

    def get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = time.time() - cache_path.stat().st_mtime
        return cache_age < self.cache_duration

    def get(self, key: str) -> Optional[Dict]:
        try:
            cache_path = self.get_cache_path(key)
            if self.is_cache_valid(cache_path):
                with open(cache_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")
        return None

    def set(self, key: str, data: Dict) -> None:
        try:
            cache_path = self.get_cache_path(key)
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error writing cache: {str(e)}")

# Initialize cache
cache = DataCache(CACHE_DIR)

def check_api_health() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        return False

@st.cache_data(ttl=3600)
def get_countries() -> List[str]:
    """Fetch available countries with proper error handling."""
    try:
        response = requests.get(f'{API_BASE_URL}/countries', timeout=10)
        response.raise_for_status()
        return response.json()['countries']
    except Exception as e:
        logger.error(f"Error fetching countries: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_symbols(csv_file: str, country: Optional[str] = None) -> List[str]:
    """
    Fetch symbols with country filtering and error handling.
    The backend is faster because it serves data from memory.
    """
    try:
        cache_key = f"symbols_{csv_file}_{country}"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return cached_data
            
        url = f'{API_BASE_URL}/csv/{csv_file}'
        if country:
            url += f'?country={country}'
            
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # The backend sends JSON directly, parse to DataFrame
        data_json = response.json()
        df = pd.DataFrame(data_json)
        
        # Assuming the first column is the symbol column
        symbols = df.iloc[:, 0].dropna().unique().tolist()
        
        cache.set(cache_key, symbols)
        return symbols
        
    except Exception as e:
        logger.error(f"Error fetching symbols: {str(e)}")
        return []

def display_model_metrics(results: Dict) -> pd.DataFrame:
    """Display model metrics in a formatted table."""
    try:
        metrics_df = pd.DataFrame([
            {
                'Model': model,
                'RMSE': details.get('rmse', np.nan),
                'MSE': details.get('mse', np.nan),
                'MAE': details.get('mae', np.nan),
                'R²': details.get('r2', np.nan),
                'MAPE (%)': details.get('mape', np.nan),
                'Status': 'Success' if 'mse' in details else 'Failed'
            }
            for model, details in results.items()
        ]).sort_values('RMSE')
        
        return metrics_df.style.format({
            'RMSE': '{:.2f}',
            'MSE': '{:.2f}',
            'MAE': '{:.2f}',
            'R²': '{:.3f}',
            'MAPE (%)': '{:.2f}'
        })
    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        return pd.DataFrame()

def plot_model_comparison(results: Dict) -> go.Figure:
    """Create a comparison plot of model performance metrics."""
    try:
        metrics = ['RMSE', 'MSE', 'MAE', 'MAPE']
        models = []
        metric_values = {metric: [] for metric in metrics}
        
        for model, details in results.items():
            if 'mse' in details:  # Only include successful models
                models.append(model)
                for metric in metrics:
                    metric_values[metric].append(details.get(metric.lower(), np.nan))
        
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=metric_values[metric],
                text=[f'{v:.2f}' for v in metric_values[metric]],
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            xaxis_title='Models',
            yaxis_title='Metric Value',
            showlegend=True
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating comparison plot: {str(e)}")
        return go.Figure()

###############################
# Basic Sentiment via TextBlob
###############################
def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Run a basic TextBlob sentiment analysis on the text.
    Returns a dict with polarity and subjectivity.
    """
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

def fetch_company_info(symbol: str) -> Dict:
    """
    Fetch basic company info using yfinance.
    E.g., ticker.info
    """
    try:
        stock = yf.Ticker(symbol)
        info_data = stock.info  # returns a dict with company info
        return info_data if info_data else {}
    except Exception as e:
        logger.error(f"Error fetching company info for {symbol}: {str(e)}")
        return {}

def main():
    try:
        st.set_page_config(
            page_title="Financial Prediction Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
            <style>
            .stAlert {
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 0.5rem;
            }
            .stDataFrame {
                padding: 1rem;
            }
            .dataframe {
                font-size: 0.9rem;
            }
            .dataframe td, .dataframe th {
                padding: 8px;
                text-align: right;
            }
            .dataframe tr:nth-child(even) {
                background-color: #f5f5f5;
            }
            .metric-container {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }
            .plot-container {
                background-color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
        """, unsafe_allow_html=True)

        st.title('📈 Advanced Financial Prediction Dashboard')
        
        # API Health Check
        if not check_api_health():
            st.error("⚠ API service is not available. Please check the connection and try again.")
            st.stop()
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            # Data Selection
            st.subheader("Data Selection")
            countries = get_countries()
            country_choice = st.selectbox(
                'Select Country',
                ['All'] + countries,
                format_func=lambda x: 'Global (No Country Specified)' if x == 'All' else x
            )
            
            csv_choice = st.selectbox(
                'Select Asset Type',
                CSV_FILES,
                format_func=lambda x: x.replace('.csv', '').title()
            )

            # Model Parameters
            st.subheader("Model Parameters")
            prediction_period = st.slider("Prediction Period (days)", 7, 365, 30)
            confidence_interval = st.slider("Confidence Interval (%)", 80, 99, 95)
            train_test_split = st.slider("Train-Test Split (%)", 50, 90, 80)
            seasonality_period = st.number_input("Seasonality Period (days)", 1, 365, 12)

        # Main content
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Symbol Selection")
            with st.spinner('Loading symbols...'):
                selected_country = None if country_choice == 'All' else country_choice
                symbols = get_symbols(csv_choice, selected_country)

            if not symbols:
                st.error("No symbols available for the selected criteria.")
                st.stop()

            symbol_choice = st.selectbox(
                'Select Symbol',
                symbols,
                index=0 if symbols else None
            )
            
            if st.button('Begin Prediction', key='predict', use_container_width=True):
                if symbol_choice:
                    with st.spinner('Training models and generating predictions...'):
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/train_model",
                                json={
                                    "symbol": symbol_choice,
                                    "period": prediction_period,
                                    "confidence": confidence_interval,
                                    "train_split": train_test_split / 100,
                                    "seasonality": seasonality_period
                                },
                                timeout=120
                            )
                            response.raise_for_status()
                            result = response.json()

                            if result['status'] == 'success':
                                st.success("Model training completed successfully! 🎉")
                                
                                # Store results in session state
                                st.session_state.prediction_results = result
                                st.session_state.selected_symbol = symbol_choice
                            else:
                                st.error(f"Model training failed: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            logger.error(f"Error during prediction: {str(e)}")

        # If we have prediction results, display them in tabs
        with col2:
            if 'prediction_results' in st.session_state:
                result = st.session_state.prediction_results
                symbol = st.session_state.selected_symbol
                
                # 5 tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Model Performance",
                    "Forecast Visualization",
                    "Analysis",
                    "Company Info",
                    "News & Sentiment"
                ])
                
                with tab1:
                    st.subheader("Model Comparison")
                    metrics_df = display_model_metrics(result['results'])
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    if result.get('best_model'):
                        st.success(f"Best performing model: {result['best_model']}")
                    
                    with st.container():
                        st.plotly_chart(plot_model_comparison(result['results']), use_container_width=True)
                
                with tab2:
                    if 'plot_url' in result:
                        try:
                            html_content = requests.get(f"http://127.0.0.1:5000{result['plot_url']}").text
                            st.components.v1.html(html_content, height=600)
                        except Exception as e:
                            st.error(f"Error loading forecast visualization: {str(e)}")
                
                with tab3:
                    st.subheader("Training Summary")
                    summary = result.get('training_summary', {})
                    
                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        st.metric("Total Data Points", summary.get('data_points', 0))
                    with metrics_cols[1]:
                        st.metric("Training Size", summary.get('train_size', 0))
                    with metrics_cols[2]:
                        st.metric("Test Size", summary.get('test_size', 0))
                    
                    # Convert decimal fraction to percent
                    seasonal_strength_pct = f"{result.get('seasonal_strength', 0) * 100:.2f}%"
                    st.metric("Seasonal Strength", seasonal_strength_pct)
                    
                    # Download results
                    st.download_button(
                        "Download Results CSV",
                        pd.DataFrame(result['results']).to_csv(index=False),
                        f"{symbol}_results.csv",
                        "text/csv",
                        use_container_width=True
                    )

                # Tab 4: Basic Company Info
                with tab4:
                    st.subheader("Company Info")

                    if st.button("Fetch Company Info", key="fetch_info"):
                        with st.spinner("Retrieving company info..."):
                            company_data = fetch_company_info(symbol)
                            
                            if not company_data:
                                st.warning("No company information found.")
                            else:
                                info_items = [{"Field": k, "Value": v} for k, v in company_data.items()]
                                info_df = pd.DataFrame(info_items)
                                st.table(info_df)

                # Tab 5: News & Sentiment
                with tab5:
                    st.subheader("News & Sentiment")

                    if st.button("Fetch Latest News", key="fetch_news"):
                        with st.spinner("Fetching news articles..."):
                            try:
                                # Use yfinance to fetch news data
                                stock = yf.Ticker(symbol)
                                news_articles = stock.news  # Use .news property
                                print(symbol)
                                print(stock)
                                print(news_articles)

                                if not news_articles:
                                    st.warning("No news articles available for this symbol.")
                                else:
                                    # Process and display the fetched news data
                                    news_rows = []
                                    for article in news_articles:
                                        # Extract fields from the article
                                        uuid = article.get("uuid", "N/A")
                                        title = article.get("title", "N/A")
                                        publisher = article.get("publisher", "Unknown Publisher")
                                        link = article.get("link", "#")
                                        provider_publish_time = article.get("providerPublishTime", "Unknown Date")
                                        type_ = article.get("type", "N/A")
                                        summary = title  # Placeholder: Use title as summary if no explicit summary is available

                                        # Analyze sentiment of the title (as summary)
                                        sentiment = analyze_sentiment(summary)

                                        # Append the processed data to the news rows
                                        news_rows.append({
                                            "UUID": uuid,
                                            "Title": title,
                                            "Publisher": publisher,
                                            "Published At": provider_publish_time,
                                            "Type": type_,
                                            "Summary": summary,
                                            "Polarity": sentiment["polarity"],
                                            "Subjectivity": sentiment["subjectivity"],
                                            "Link": link
                                        })

                                    # Create a DataFrame for displaying news and sentiment analysis
                                    news_df = pd.DataFrame(news_rows)

                                    # Display a clean table with sentiment scores
                                    st.write("### News Articles with Sentiment Analysis")
                                    st.dataframe(news_df.drop(columns=["Link"]), use_container_width=True)

                                    # Display interactive news links
                                    st.write("### News Links")
                                    for _, row in news_df.iterrows():
                                        st.markdown(f"- [{row['Title']}]({row['Link']}) by {row['Publisher']} (Published At: {row['Published At']})")

                            except Exception as e:
                                logger.error(f"Error retrieving news for {symbol}: {str(e)}")
                                st.error(f"Error fetching news articles: {str(e)}")

        # Footer
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align: center'>
                <p>Made with ❤ | Data provided by Yahoo Finance | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")

    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        st.error("An unexpected error occurred. Please try again or contact support.")

if __name__ == "__main__":
    main()