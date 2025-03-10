import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load Dataset
df = pd.read_csv('data.csv')

# Function to identify missing data type
def identify_missing_data(df):
    """
    Identifies the type of missing data mechanism:
    - MCAR (Missing Completely At Random)
    - MAR (Missing At Random)
    - MNAR (Missing Not At Random)
    
    Returns the most likely mechanism based on tests.
    """
    # First, create missing indicators for all columns with missing values
    missing_indicators = {}
    missing_columns = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_columns.append(col)
            missing_indicators[col] = df[col].isnull().astype(int)
    
    if not missing_columns:
        return "No Missing Data"
    
    # Test for MCAR: Are missing patterns independent of each other?
    # Simple approximation of Little's MCAR test
    missing_df = pd.DataFrame(missing_indicators)
    if len(missing_columns) > 1:
        contingency_tables = []
        for i, col1 in enumerate(missing_columns[:-1]):
            for col2 in missing_columns[i+1:]:
                table = pd.crosstab(missing_df[col1], missing_df[col2])
                if min(table.shape) > 1:  # Only test if we have a valid contingency table
                    chi2, p, _, _ = chi2_contingency(table)
                    contingency_tables.append(p)
        
        # If all p-values are high, missing data patterns may be independent (MCAR)
        if contingency_tables and all(p > 0.05 for p in contingency_tables):
            return "MCAR"
    
    # Test for MAR: Is missingness correlated with observed values in other columns?
    mar_evidence = False
    
    for col in missing_columns:
        # Get other columns that might predict missingness
        other_cols = [c for c in df.columns if c != col]
        non_missing_df = df[other_cols].copy()
        
        # Add the missingness indicator
        non_missing_df['is_missing'] = missing_indicators[col]
        
        # Only include numeric columns for correlation analysis
        numeric_cols = non_missing_df.select_dtypes(include=['number']).columns.tolist()
        if 'is_missing' in numeric_cols and len(numeric_cols) > 1:
            # Calculate correlation of missingness with other numeric variables
            correlations = non_missing_df[numeric_cols].corr()['is_missing'].drop('is_missing')
            if any(abs(corr) > 0.2 for corr in correlations):
                mar_evidence = True
                break
    
    if mar_evidence:
        return "MAR"
    
    # If neither MCAR nor MAR is evident, default to MNAR
    # (the most conservative assumption)
    return "MNAR"

# Handling Numerical Missing Values with Simple Imputation
def handle_numerical_missing(df, method='auto'):
    """
    Imputes numerical missing values using mean or median based on data skew.
    """
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['number']).columns:
        if df_copy[col].isnull().sum() > 0:
            if method == 'auto':
                # Use median for skewed data, mean for normally distributed data
                strategy = 'mean' if abs(df_copy[col].skew()) < 1 else 'median'
            else:
                strategy = method
            imputer = SimpleImputer(strategy=strategy)
            df_copy[col] = imputer.fit_transform(df_copy[[col]])
    return df_copy

# KNN Imputation (For MAR)
def knn_impute(df, n_neighbors=5):
    """
    Imputes missing values using K-Nearest Neighbors approach.
    Good for MAR data where relationships between variables can help prediction.
    """
    df_copy = df.copy()
    
    # Handle categorical variables first (KNN works on numeric data)
    df_copy = handle_categorical_missing(df_copy, 'mode')
    
    # Create dummy variables for remaining categorical columns
    categorical_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df_copy = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)
    
    # Apply KNN imputation to numeric data
    numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
    
    return df_copy

# MICE (Multiple Imputation by Chained Equations) - Better for MNAR
def mice_imputation(df, max_iter=10):
    """
    Performs multiple imputation using MICE (implemented via sklearn's IterativeImputer).
    Good for complex missing patterns including MNAR.
    """
    df_copy = df.copy()
    
    # First handle categorical variables
    df_copy = handle_categorical_missing(df_copy, 'mode')
    
    # Create dummy variables for remaining categorical columns
    categorical_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df_copy = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)
    
    # Apply MICE to numeric data
    numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        # Using Random Forest as the estimator for better handling of non-linear relationships
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=50),
            max_iter=max_iter, 
            random_state=42
        )
        df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
    
    return df_copy

# Handling Categorical Missing Values
def handle_categorical_missing(df, method='mode'):
    """
    Imputes categorical missing values using either mode or probabilistic sampling.
    """
    df_copy = df.copy()
    
    for col in df_copy.select_dtypes(include=['object']).columns:
        if df_copy[col].isnull().sum() > 0:
            if method == 'mode':
                # Most frequent value imputation
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            elif method == 'probabilistic':
                # Probabilistic imputation based on observed distribution
                probabilities = df_copy[col].value_counts(normalize=True)
                missing_indices = df_copy[col].isnull()
                
                # Generate random values according to the observed distribution
                random_values = np.random.choice(
                    probabilities.index, 
                    size=missing_indices.sum(), 
                    p=probabilities.values
                )
                
                # Assign the random values to missing spots
                df_copy.loc[missing_indices, col] = random_values
    
    return df_copy

# Create missing value indicators (useful for MNAR)
def add_missing_indicators(df):
    """
    Adds binary indicator columns for each feature with missing values.
    This preserves information about the missingness pattern.
    """
    df_copy = df.copy()
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df_copy[f'{col}_missing'] = df[col].isnull().astype(int)
    
    return df_copy

# Drop High-Missing Features
def drop_high_missing_features(df, threshold=50):
    """
    Drops features with missing percentage above the threshold.
    """
    df_copy = df.copy()
    
    missing_percentage = df.isnull().sum() * 100 / len(df)
    drop_columns = missing_percentage[missing_percentage > threshold].index
    
    if not drop_columns.empty:
        df_copy = df_copy.drop(columns=drop_columns)
        print(f"Dropped columns with >{threshold}% missing values: {list(drop_columns)}")
    
    return df_copy

# Complete Pipeline Execution
def process_missing_data(df, identify_mechanism=True, drop_threshold=50, preserve_missingness=True):
    """
    Complete pipeline for handling missing data with appropriate techniques.
    
    Parameters:
    - df: Input DataFrame
    - identify_mechanism: Whether to automatically identify the missing data mechanism
    - drop_threshold: Percentage threshold for dropping high-missing columns
    - preserve_missingness: Whether to add missing indicators
    
    Returns:
    - Processed DataFrame with imputed values
    """
    print("Starting missing data processing pipeline...")
    
    # First, drop columns with too many missing values
    df_processed = drop_high_missing_features(df, threshold=drop_threshold)
    
    # Optionally preserve missingness patterns
    if preserve_missingness:
        df_processed = add_missing_indicators(df_processed)
    
    # Identify missing data mechanism
    if identify_mechanism:
        missing_type = identify_missing_data(df_processed)
        print(f"Identified missing data mechanism: {missing_type}")
    else:
        # Default to MNAR as the most conservative approach
        missing_type = "MNAR"
        print(f"Using default missing data mechanism: {missing_type}")
    
    # Apply appropriate imputation strategy
    if missing_type == "MCAR":
        df_processed = handle_numerical_missing(df_processed)
        df_processed = handle_categorical_missing(df_processed, method='mode')
        print("Applied simple imputation (mean/median for numeric, mode for categorical)")
    
    elif missing_type == "MAR":
        df_processed = knn_impute(df_processed)
        print("Applied KNN imputation for MAR pattern")
    
    elif missing_type == "MNAR":
        df_processed = mice_imputation(df_processed)
        print("Applied MICE imputation for MNAR pattern")
    
    # Verify no missing values remain
    remaining_missing = df_processed.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values remain. Applying fallback imputation.")
        df_processed = handle_numerical_missing(df_processed, method='median')
        df_processed = handle_categorical_missing(df_processed, method='mode')
    
    print("Missing data processing complete.")
    return df_processed

# Execute the pipeline
processed_df = process_missing_data(df)

# Save Processed Data
processed_df.to_csv('processed_data.csv', index=False)