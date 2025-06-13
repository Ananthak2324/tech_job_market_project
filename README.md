# Tech Industry Layoff Analysis & Prediction System

This project analyzes tech industry layoffs and provides predictions for future trends using machine learning. It includes an interactive dashboard for visualizing the data and predictions.

## Features

- Machine learning model for predicting layoff trends
- Interactive Streamlit dashboard
- PostgreSQL database integration
- Data visualization and analysis tools

## Deployment Instructions

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env` file
4. Run the dashboard:
   ```bash
   streamlit run scripts/dashboard.py
   ```

### Streamlit Cloud Deployment

1. Create a Streamlit Cloud account at https://streamlit.io/cloud
2. Connect your GitHub repository
3. Deploy the app using the dashboard.py file
4. Share the generated public URL

## Project Structure

- `scripts/`: Contains Python scripts for data processing and analysis
- `visualizations/`: Stores generated plots and charts
- `.streamlit/`: Streamlit configuration files
- `requirements.txt`: Project dependencies

## Environment Variables

Create a `.env` file with the following variables:
```
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=tech_layoffs
```

## License

MIT License
