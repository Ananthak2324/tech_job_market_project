import pandas as pd
import numpy as np

# Load the raw CSV
df = pd.read_csv("data/raw/tech_layoffs.csv")

# Remove the '#' column if present
if '#' in df.columns:
    df = df.drop(columns=['#'])

# Clean numeric columns (convert to int where appropriate)
int_cols = [
    'Laid_Off', 'Company_Size_before_Layoffs', 'Company_Size_after_layoffs', 'Year'
]
for col in int_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Clean float columns
float_cols = ['Percentage', 'lat', 'lng']
for col in float_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean Money_Raised_in_$_mil (remove $ and commas, convert to float)
df['Money_Raised_in_$_mil'] = (
    df['Money_Raised_in_$_mil']
    .replace('[\$,]', '', regex=True)
    .replace('', np.nan)
    .replace('nan', np.nan)
    .astype(float)
)

# Convert Date_layoffs to date string (YYYY-MM-DD)
df['Date_layoffs'] = pd.to_datetime(df['Date_layoffs'], errors='coerce').dt.strftime('%Y-%m-%d')

# Save cleaned CSV
df.to_csv("data/raw/tech_layoffs_cleaned.csv", index=False)
print("Cleaned CSV saved as data/raw/tech_layoffs_cleaned.csv") 