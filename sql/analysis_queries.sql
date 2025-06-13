-- Verify data loading
SELECT COUNT(*) as total_records FROM tech_layoffs;

-- Monthly layoff trends
SELECT 
    DATE_TRUNC('month', date_layoffs) as month,
    SUM(laid_off) as total_layoffs,
    COUNT(DISTINCT company) as companies_affected
FROM tech_layoffs
GROUP BY DATE_TRUNC('month', date_layoffs)
ORDER BY month;

-- Top 10 companies by layoff size
SELECT 
    company,
    SUM(laid_off) as total_layoffs,
    COUNT(*) as number_of_events,
    AVG(laid_off) as avg_layoff_size
FROM tech_layoffs
GROUP BY company
ORDER BY total_layoffs DESC
LIMIT 10;

-- Industry analysis
SELECT 
    industry,
    SUM(laid_off) as total_layoffs,
    COUNT(DISTINCT company) as companies_affected,
    AVG(laid_off) as avg_layoff_size
FROM tech_layoffs
GROUP BY industry
ORDER BY total_layoffs DESC;

-- Geographic distribution
SELECT 
    country,
    SUM(laid_off) as total_layoffs,
    COUNT(DISTINCT company) as companies_affected
FROM tech_layoffs
GROUP BY country
ORDER BY total_layoffs DESC; 