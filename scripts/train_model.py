import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database connection
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Create database connection
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

def load_data():
    """Load data from PostgreSQL database"""
    query = """
    SELECT 
        "Date_layoffs" as date_layoffs,
        company,
        industry,
        country,
        laid_off,
        percentage,
        money_raised_in_$_mil,
        stage
    FROM tech_layoffs
    WHERE "Date_layoffs" IS NOT NULL
    AND industry IS NOT NULL
    """
    return pd.read_sql(query, engine)

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Convert date to datetime
    df['date_layoffs'] = pd.to_datetime(df['date_layoffs'])
    
    # Create time-based features
    df['year'] = df['date_layoffs'].dt.year
    df['month'] = df['date_layoffs'].dt.month
    df['quarter'] = df['date_layoffs'].dt.quarter
    
    # Create monthly aggregates by industry
    monthly_data = df.groupby(['year', 'month', 'industry']).agg({
        'laid_off': 'sum',
        'company': 'count'
    }).reset_index()
    
    monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
    
    return monthly_data

def create_features(df):
    """Create features for the model"""
    # Create lag features by industry
    df['laid_off_lag_1'] = df.groupby('industry')['laid_off'].shift(1)
    df['laid_off_lag_2'] = df.groupby('industry')['laid_off'].shift(2)
    df['laid_off_lag_3'] = df.groupby('industry')['laid_off'].shift(3)
    
    # Create rolling mean features by industry
    df['laid_off_rolling_mean_3'] = df.groupby('industry')['laid_off'].transform(lambda x: x.rolling(window=3).mean())
    df['laid_off_rolling_mean_6'] = df.groupby('industry')['laid_off'].transform(lambda x: x.rolling(window=6).mean())
    
    # Add industry-specific features
    industry_means = df.groupby('industry')['laid_off'].mean().reset_index()
    industry_means.columns = ['industry', 'industry_mean']
    df = df.merge(industry_means, on='industry', how='left')
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def train_model(df):
    """Train the machine learning model"""
    # Prepare features and target
    features = ['laid_off_lag_1', 'laid_off_lag_2', 'laid_off_lag_3',
                'laid_off_rolling_mean_3', 'laid_off_rolling_mean_6',
                'industry_mean']
    X = df[features]
    y = df['laid_off']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    print(f"Root Mean Square Error: {rmse:,.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_mean_6', 'industry_mean'],
        'importance': model.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))

def plot_predictions(df, model, scaler):
    """Plot actual vs predicted values by industry"""
    # Prepare features for prediction
    features = ['laid_off_lag_1', 'laid_off_lag_2', 'laid_off_lag_3',
                'laid_off_rolling_mean_3', 'laid_off_rolling_mean_6',
                'industry_mean']
    X = df[features]
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Plot for each industry
    top_industries = df.groupby('industry')['laid_off'].sum().nlargest(5).index
    
    plt.figure(figsize=(15, 10))
    for i, industry in enumerate(top_industries, 1):
        industry_data = df[df['industry'] == industry]
        industry_preds = predictions[df['industry'] == industry]
        
        plt.subplot(3, 2, i)
        plt.plot(industry_data['date'], industry_data['laid_off'], label='Actual', color='blue')
        plt.plot(industry_data['date'], industry_preds, label='Predicted', color='red', linestyle='--')
        plt.title(f'{industry} Layoffs')
        plt.xlabel('Date')
        plt.ylabel('Number of Layoffs')
        plt.legend()
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/industry_layoff_predictions.png')
    plt.close()
    
    # Create a summary plot of all industries
    plt.figure(figsize=(12, 6))
    total_actual = df.groupby('date')['laid_off'].sum()
    total_pred = pd.Series(predictions, index=df['date']).groupby('date').sum()
    
    plt.plot(total_actual.index, total_actual.values, label='Actual Total', color='blue')
    plt.plot(total_pred.index, total_pred.values, label='Predicted Total', color='red', linestyle='--')
    plt.title('Total Tech Industry Layoffs: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Number of Layoffs')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/total_layoff_predictions.png')
    plt.close()

def main():
    # Load and preprocess data
    print("Loading data...")
    df = load_data()
    monthly_data = preprocess_data(df)
    
    # Create features
    print("Creating features...")
    feature_data = create_features(monthly_data)
    
    # Train model
    print("Training model...")
    model, scaler, X_test, y_test = train_model(feature_data)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Plot predictions
    print("Generating plots...")
    plot_predictions(feature_data, model, scaler)
    
    print("Analysis complete! Check visualizations/ directory for plots.")

if __name__ == "__main__":
    main()
