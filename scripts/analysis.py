import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TechLayoffsAnalyzer:
    def __init__(self):
        self.data_path = os.path.join('data', 'processed', 'tech_layoffs_cleaned.csv')
        self.df = None
        self.engine = None
        
    def load_data(self):
        """Load data from CSV and setup database connection"""
        # Load CSV data
        self.df = pd.read_csv(self.data_path)
        self.df['date_layoffs'] = pd.to_datetime(self.df['date_layoffs'])
        
        # Setup database connection
        try:
            self.engine = create_engine(os.getenv('DATABASE_URL'))
        except:
            print("Database connection not configured. Running in CSV-only mode.")
    
    def analyze_temporal_trends(self):
        """Analyze layoff trends over time"""
        # Monthly trends
        monthly_layoffs = self.df.groupby(self.df['date_layoffs'].dt.to_period('M'))['laid_off'].sum()
        
        # Yearly trends
        yearly_layoffs = self.df.groupby(self.df['date_layoffs'].dt.year)['laid_off'].sum()
        
        # Plot monthly trends
        plt.figure(figsize=(15, 6))
        monthly_layoffs.plot(kind='line')
        plt.title('Monthly Tech Layoffs Over Time')
        plt.xlabel('Month')
        plt.ylabel('Number of Layoffs')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('monthly_trends.png')
        plt.close()
        
        return monthly_layoffs, yearly_layoffs
    
    def analyze_geographic_distribution(self):
        """Analyze layoffs by geographic region"""
        # Country analysis
        country_stats = self.df.groupby('country').agg({
            'laid_off': ['sum', 'count'],
            'company': 'nunique'
        }).round(2)
        
        country_stats.columns = ['total_layoffs', 'number_of_events', 'unique_companies']
        country_stats = country_stats.sort_values('total_layoffs', ascending=False)
        
        # Plot top 10 countries
        plt.figure(figsize=(12, 6))
        country_stats.head(10)['total_layoffs'].plot(kind='bar')
        plt.title('Top 10 Countries by Total Layoffs')
        plt.xlabel('Country')
        plt.ylabel('Total Layoffs')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('top_countries.png')
        plt.close()
        
        return country_stats
    
    def analyze_industry_trends(self):
        """Analyze layoffs by industry"""
        industry_stats = self.df.groupby('industry').agg({
            'laid_off': ['sum', 'count'],
            'company': 'nunique'
        }).round(2)
        
        industry_stats.columns = ['total_layoffs', 'number_of_events', 'unique_companies']
        industry_stats = industry_stats.sort_values('total_layoffs', ascending=False)
        
        # Plot top 10 industries
        plt.figure(figsize=(12, 6))
        industry_stats.head(10)['total_layoffs'].plot(kind='bar')
        plt.title('Top 10 Industries by Total Layoffs')
        plt.xlabel('Industry')
        plt.ylabel('Total Layoffs')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('top_industries.png')
        plt.close()
        
        return industry_stats
    
    def prepare_for_ml(self):
        """Prepare data for machine learning"""
        # Create time-based features
        self.df['year'] = self.df['date_layoffs'].dt.year
        self.df['month'] = self.df['date_layoffs'].dt.month
        self.df['quarter'] = self.df['date_layoffs'].dt.quarter
        
        # Create aggregated features
        company_stats = self.df.groupby('company').agg({
            'laid_off': ['sum', 'count', 'mean'],
            'date_layoffs': ['min', 'max']
        }).round(2)
        
        company_stats.columns = ['total_layoffs', 'layoff_events', 'avg_layoff_size',
                               'first_layoff', 'last_layoff']
        
        # Save processed data
        company_stats.to_csv('data/processed/company_stats.csv')
        
        return company_stats
    
    def run_analysis(self):
        """Run all analyses"""
        print("Loading data...")
        self.load_data()
        
        print("\nAnalyzing temporal trends...")
        monthly_trends, yearly_trends = self.analyze_temporal_trends()
        
        print("\nAnalyzing geographic distribution...")
        country_stats = self.analyze_geographic_distribution()
        
        print("\nAnalyzing industry trends...")
        industry_stats = self.analyze_industry_trends()
        
        print("\nPreparing data for machine learning...")
        company_stats = self.prepare_for_ml()
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Total number of layoff events: {len(self.df)}")
        print(f"Total number of people laid off: {self.df['laid_off'].sum():,.0f}")
        print(f"Date range: {self.df['date_layoffs'].min()} to {self.df['date_layoffs'].max()}")
        print(f"Number of companies affected: {self.df['company'].nunique()}")
        print(f"Number of countries affected: {self.df['country'].nunique()}")
        
        return {
            'monthly_trends': monthly_trends,
            'yearly_trends': yearly_trends,
            'country_stats': country_stats,
            'industry_stats': industry_stats,
            'company_stats': company_stats
        }

if __name__ == "__main__":
    analyzer = TechLayoffsAnalyzer()
    results = analyzer.run_analysis() 