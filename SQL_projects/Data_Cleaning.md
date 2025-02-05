# Data Cleaning

The `Data_Cleaning.sql` script provides functionality to clean and standardize data in a table named `layoffs`. It includes steps to remove duplicates, standardize data, handle null or blank values, and remove unnecessary columns.

## Description

The script performs the following operations:
1. Removes duplicates from the `layoffs_staging` table based on specific columns.
2. Standardizes data by trimming whitespace, updating industry names, and converting date formats.
3. Handles null or blank values in the `industry` column.
4. Removes any rows where both `total_laid_off` and `percentage_laid_off` are null.
5. Drops the temporary column `row_num`.

## Usage Examples

### Remove Duplicates
```sql
-- Remove duplicates from layoffs_staging
WITH duplicate_cte AS (
    SELECT *,
    ROW_NUMBER() OVER(PARTITION BY company, location, industry, total_laid_off, percentage_laid_off, `date`, stage, country, funds_raised_millions) AS row_num
    FROM layoffs_staging
)
SELECT *
FROM duplicate_cte
WHERE row_num > 1;
```

### Standardize the Data
```sql
-- Trim whitespace in company names
UPDATE layoffs_staging2
SET company = TRIM(company);

-- Update industry names to 'Crypto' for those starting with 'Crypto%'
UPDATE layoffs_staging2
SET industry = 'Crypto'
WHERE industry LIKE 'Crypto%';
```

### Null Values / Blank Values
```sql
-- Handle NULL or empty values in the industry column for Airbnb company
UPDATE layoffs_staging2
SET industry = NULL
WHERE company = 'Airbnb';
```

### Remove Any Columns
```sql
-- Remove rows where total_laid_off and percentage_laid_off are both NULL 
DELETE FROM layoffs_staging2 WHERE total_laid_off IS NULL AND percentage_laid_off IS NULL;
```

## Additional Notes

- The script uses common SQL commands like SELECT, INSERT INTO, UPDATE, DELETE to manipulate data.
- Temporary tables (`layoffs_staging`, `layoffs_staging2`) are created to perform cleaning operations without affecting the original data.
- The script demonstrates how to use CTEs (Common Table Expressions) for removing duplicates efficiently.

By following these steps outlined in the script, you can clean and standardize your data effectively.

Feel free to modify the script according to your specific requirements or integrate it into your ETL (Extract Transform Load) processes for data cleaning tasks.