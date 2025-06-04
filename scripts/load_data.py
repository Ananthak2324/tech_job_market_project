import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv("DB_URL") + "?client_encoding=utf8")

df = pd.read_excel("../data/raw/tech_layoffs.xlsx")

df.columns = [col.strip().lower().replace(" ", "_").replace("#", "id") for col in df.columns]

df['money_raised_in_$_mil'] = (
    df['money_raised_in_$_mil']
    .replace(r'[\$,]', '', regex=True)
    .astype(float)
)
df['date_layoffs'] = pd.to_datetime(df['date_layoffs'], errors='coerce')
df.rename(columns={
    'company_size_before_layoffs': 'company_size_before',
    'company_size_after_layoffs': 'company_size_after',
    'money_raised_in_$_mil': 'money_raised_mil'
}, inplace=True)

df = df[df['company'].notna()]

df.to_sql("tech_layoffs", engine, if_exists="replace", index=False)

print("Data successfully loaded into PostgreSQL.")
