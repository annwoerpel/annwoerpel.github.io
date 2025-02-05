# Exploratory Data Analysis

## Description
This SQL script performs exploratory data analysis on the `layoffs_staging2` table. It includes various queries to analyze different aspects of the data such as total layoffs, percentages, companies, industries, countries, dates, and more.

## Usage Examples
1. Retrieve all data from the `layoffs_staging2` table:
```sql
SELECT *
FROM layoffs_staging2;
```

2. Find the maximum values of total laid off and percentage laid off:
```sql
SELECT MAX(total_laid_off), MAX(percentage_laid_off)
FROM layoffs_staging2;
```

3. Filter and order data where percentage laid off is 1:
```sql
SELECT *
FROM layoffs_staging2
WHERE percentage_laid_off = 1
ORDER BY total_laid_off DESC;
```

4. Group by company and order by total laid off in descending order:
```sql
SELECT company, SUM(total_laid_off)
FROM layoffs_staging2
GROUP BY company
ORDER BY 2 DESC;
```

5. Calculate rolling totals of total laid off per month:
```sql
WITH Rolling_Total AS 
(
    SELECT SUBSTRING(`date`, 1, 7) AS `MONTH`, SUM(total_laid_off) AS sum_tlo
    FROM layoffs_staging2
    WHERE SUBSTRING(`date`, 1, 7) IS NOT NULL
    GROUP BY `MONTH`
    ORDER BY 1 ASC
)
SELECT `MONTH`, sum_tlo, SUM(sum_tlo) OVER(ORDER BY `MONTH`) AS rolling_total
FROM Rolling_Total;

## Parameters

- No parameters are passed to these SQL queries.

## Return Values

- Each query returns specific information based on the analysis being performed.
- The result sets may vary depending on the data in the `layoffs_staging2` table.

## Additional Notes

- This script provides a comprehensive analysis of the data in the `layoffs_staging2` table.
- Various aggregations and filters are applied to gain insights into different aspects of the dataset.
- The script includes calculations for averages, sums, rankings, and rolling totals.
- Modify the queries as needed to suit your specific analytical requirements or adapt them for use with other datasets.