-- Test Query to check and see if table prints first 50 rows of tech_layoffs
SELECT * FROM tech_layoffs LIMIT 50;

-- Basic Queries to see different aspects of the data

-- 1. Total layoffs per year
SELECT 
    EXTRACT(YEAR FROM date_layoffs) AS year, COUNT(*) AS total_layoffs
FROM tech_layoffs
GROUP BY EXTRACT(YEAR FROM date_layoffs)
ORDER BY year;

-- 2. Companies with the most layoff events
SELECT 
    company, 
    COUNT(*) AS layoff_events
FROM tech_layoffs
GROUP BY company
ORDER BY layoff_events DESC
LIMIT 10;

-- 3. Layoffs by country
SELECT 
    country,
    COUNT(*) AS total_layoffs
FROM tech_layoffs
GROUP BY country
ORDER BY total_layoffs DESC;

SELECT * FROM tech_layoffs LIMIT 1;
