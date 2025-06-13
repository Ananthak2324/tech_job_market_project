import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sqlalchemy import create_engine, text
from config.database import DB_CONFIG, DATABASE_URL

def setup_database():
    """Set up the database and create necessary tables"""
    # Step 1: Connect to the default 'postgres' database to create 'tech_layoffs_db' if it doesn't exist
    default_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/postgres"
    default_engine = create_engine(default_url)
    try:
        with default_engine.connect() as conn:
            conn.execute(text("commit"))
            conn.execute(text("CREATE DATABASE tech_layoffs_db"))
            print("Database 'tech_layoffs_db' created.")
    except Exception as e:
        print(f"Database might already exist: {e}")

    # Step 2: Connect to the new 'tech_layoffs_db' database
    engine = create_engine(DATABASE_URL)
    
    # Read the CSV file with utf-8 encoding
    df = pd.read_csv('data/processed/tech_layoffs_cleaned.csv', encoding='utf-8')
    
    # Convert date column to datetime
    df['date_layoffs'] = pd.to_datetime(df['date_layoffs'])
    
    # Create table and load data with method='multi' for better Unicode support
    df.to_sql('tech_layoffs', engine, if_exists='replace', index=False, method='multi')
    
    print("Database setup complete!")
    print(f"Loaded {len(df)} records into the database.")

if __name__ == "__main__":
    setup_database() 