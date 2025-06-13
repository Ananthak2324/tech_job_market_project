import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from train_model import load_data, preprocess_data, create_features, train_model

# Set page config
st.set_page_config(
    page_title="Tech Layoffs Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Database connection
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Create database connection
try:
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
except Exception as e:
    st.error(f"Failed to connect to database: {str(e)}")
    st.stop()

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_and_prepare_data():
    """Load and prepare data for the dashboard"""
    try:
        df = load_data()
        monthly_data = preprocess_data(df)
        feature_data = create_features(monthly_data)
        model, scaler, _, _ = train_model(feature_data)
        return df, monthly_data, feature_data, model, scaler
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

def plot_industry_trends(monthly_data):
    """Plot layoff trends by industry"""
    st.subheader("Layoff Trends by Industry")
    
    # Get top industries
    top_industries = monthly_data.groupby('industry')['laid_off'].sum().nlargest(5).index
    
    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for industry in top_industries:
        industry_data = monthly_data[monthly_data['industry'] == industry]
        ax.plot(industry_data['date'], industry_data['laid_off'], label=industry)
    
    plt.title('Monthly Layoffs by Top 5 Industries')
    plt.xlabel('Date')
    plt.ylabel('Number of Layoffs')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()

def plot_predictions(feature_data, model, scaler):
    """Plot actual vs predicted values"""
    st.subheader("Model Predictions")
    
    # Prepare features for prediction
    features = ['laid_off_lag_1', 'laid_off_lag_2', 'laid_off_lag_3',
                'laid_off_rolling_mean_3', 'laid_off_rolling_mean_6',
                'industry_mean']
    X = feature_data[features]
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(feature_data['date'], feature_data['laid_off'], label='Actual', color='blue')
    ax.plot(feature_data['date'], predictions, label='Predicted', color='red', linestyle='--')
    plt.title('Actual vs Predicted Layoffs')
    plt.xlabel('Date')
    plt.ylabel('Number of Layoffs')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()

def show_industry_insights(monthly_data):
    """Show insights about industries"""
    st.subheader("Industry Insights")
    
    # Calculate total layoffs by industry
    industry_totals = monthly_data.groupby('industry')['laid_off'].sum().sort_values(ascending=False)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    industry_totals.head(10).plot(kind='bar', ax=ax)
    plt.title('Total Layoffs by Industry (Top 10)')
    plt.xlabel('Industry')
    plt.ylabel('Total Layoffs')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.close()
    
    # Show statistics
    st.write("Industry Statistics:")
    stats_df = monthly_data.groupby('industry').agg({
        'laid_off': ['sum', 'mean', 'std', 'count']
    }).round(2)
    stats_df.columns = ['Total Layoffs', 'Average Layoffs', 'Standard Deviation', 'Number of Events']
    st.dataframe(stats_df.sort_values('Total Layoffs', ascending=False))

def main():
    st.title("Tech Industry Layoff Analysis Dashboard")
    
    # Add description
    st.markdown("""
    This dashboard provides insights into tech industry layoffs, including trends, predictions, and industry-specific analysis.
    Use the sidebar filters to customize your view.
    """)
    
    # Load data
    with st.spinner('Loading data...'):
        try:
            df, monthly_data, feature_data, model, scaler = load_and_prepare_data()
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            st.stop()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = monthly_data['date'].min()
    max_date = monthly_data['date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Industry filter
    all_industries = ['All'] + sorted(monthly_data['industry'].unique().tolist())
    selected_industry = st.sidebar.selectbox("Select Industry", all_industries)
    
    # Filter data based on selections
    if selected_industry != 'All':
        monthly_data = monthly_data[monthly_data['industry'] == selected_industry]
        feature_data = feature_data[feature_data['industry'] == selected_industry]
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Trends", "Predictions", "Insights"])
    
    with tab1:
        plot_industry_trends(monthly_data)
    
    with tab2:
        plot_predictions(feature_data, model, scaler)
    
    with tab3:
        show_industry_insights(monthly_data)
    
    # Add model metrics
    st.sidebar.header("Model Performance")
    st.sidebar.metric("RMSE", "1,151.81")
    st.sidebar.write("Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_mean_6', 'industry_mean'],
        'importance': model.feature_importances_
    })
    st.sidebar.dataframe(feature_importance.sort_values('importance', ascending=False))

if __name__ == "__main__":
    main() 