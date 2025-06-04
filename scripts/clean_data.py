# scripts/clean_data.py

import pandas as pd
import os

# Load raw data
raw_path = os.path.join("..", "data", "raw", "tech_layoffs.xlsx")
df = pd.read_excel(raw_path)

# Clean and normalize column names first
df.columns = df.columns.str.strip()
df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

# ---------- Cleaning ----------

# Drop rows with missing essential fields
df = df.dropna(subset=["date_layoffs", "laid_off"])

# Convert date column
df["date_layoffs"] = pd.to_datetime(df["date_layoffs"], errors="coerce")

# Create year and month features
df["year"] = df["date_layoffs"].dt.year
df["month"] = df["date_layoffs"].dt.month

# Convert monetary columns if they exist
if "funds_raised" in df.columns:
    df["funds_raised"] = df["funds_raised"].replace(r'[\$,]', '', regex=True).astype(float)

# Clean percentage column if present
if "percentage_laid_off" in df.columns:
    df["percentage_laid_off"] = df["percentage_laid_off"].str.rstrip('%').astype(float)

# Drop irrelevant columns if they exist
drop_cols = ["source_url", "notes"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# ---------- Save Cleaned Data ----------
processed_path = os.path.join("..", "data", "processed", "tech_layoffs_cleaned.csv")
df.to_csv(processed_path, index=False)

print("✅ Cleaned data saved to:", processed_path)

