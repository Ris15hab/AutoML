import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


class AutoMLPreprocessor:
    """
    AutoML feature preprocessing module with adaptive scaling selection
    and intelligent categorical feature handling.
    """
    
    def __init__(self, 
                 skew_threshold=0.5,
                 kurt_threshold=1.0,
                 outlier_threshold=0.05,
                 cardinality_threshold=10,
                 max_categories=20):
        """
        Initialize the preprocessor with configurable thresholds.
        
        Parameters:
        -----------
        skew_threshold : float
            Threshold for absolute skewness to consider a distribution as non-normal
        kurt_threshold : float
            Threshold for absolute difference from kurtosis=3 to consider a distribution as non-normal
        outlier_threshold : float
            Threshold for proportion of outliers to trigger robust scaling
        cardinality_threshold : int
            Minimum number of unique values to consider a column as high-cardinality
        max_categories : int
            Maximum number of categories to preserve for high-cardinality features
        """
        self.skew_threshold = skew_threshold
        self.kurt_threshold = kurt_threshold
        self.outlier_threshold = outlier_threshold
        self.cardinality_threshold = cardinality_threshold
        self.max_categories = max_categories
        
        # Will store column information after fitting
        self.numerical_columns = []
        self.categorical_columns = []
        self.scaling_decisions = {}
        self.categorical_mappings = {}
        self.column_stats = {}
        
        # Transformers that will be created during fit
        self.numerical_transformer = None
        self.categorical_transformer = None
        self.preprocessor = None
    
    def _detect_outliers(self, series):
        """
        Detect proportion of outliers in a numerical series using IQR method.
        
        Parameters:
        -----------
        series : pandas.Series
            Numerical data to analyze
            
        Returns:
        --------
        float : Proportion of values that are outliers
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        return outliers / len(series)
    
    def _analyze_distribution(self, series):
        """
        Analyze the distribution characteristics of a numerical series.
        
        Parameters:
        -----------
        series : pandas.Series
            Numerical data to analyze
            
        Returns:
        --------
        dict : Statistical properties of the distribution
        """
        # Remove NaN values for analysis
        clean_series = series.dropna()
        
        if len(clean_series) < 2:
            return {
                'skewness': 0,
                'kurtosis': 0,
                'outlier_ratio': 0,
                'scaling_method': 'min_max'
            }
        
        # Calculate distribution statistics
        skewness = stats.skew(clean_series)
        kurtosis = stats.kurtosis(clean_series, fisher=False)  # Using Pearson's definition (normal = 3.0)
        outlier_ratio = self._detect_outliers(clean_series)
        
        # Determine appropriate scaling method
        scaling_method = 'standard'  # default
        
        if outlier_ratio > self.outlier_threshold:
            scaling_method = 'robust'
        elif abs(skewness) < self.skew_threshold and abs(kurtosis - 3) < self.kurt_threshold:
            scaling_method = 'standard'
        else:
            scaling_method = 'min_max'
            
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'outlier_ratio': outlier_ratio,
            'scaling_method': scaling_method
        }
    
    def _detect_categorical_columns(self, df):
        """
        Automatically detect categorical columns using multiple heuristics.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataset
            
        Returns:
        --------
        list : Column names that should be treated as categorical
        """
        categorical_columns = []
        
        for col in df.columns:
            # Skip columns with too many missing values
            if df[col].isna().mean() > 0.5:
                continue
                
            # Check data type (object/string types are likely categorical)
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                categorical_columns.append(col)
                continue
                
            # For numeric columns, check cardinality relative to dataset size
            n_unique = df[col].nunique()
            if n_unique < min(self.cardinality_threshold, len(df) * 0.05):
                categorical_columns.append(col)
                continue
                
            # Check for structured non-numeric patterns
            if df[col].dtype == 'int64' or df[col].dtype == 'int32':
                # Check if values are codes rather than measurements
                # (e.g., all integers with small range)
                if n_unique <= 20 and df[col].min() >= 0 and df[col].max() <= 100:
                    categorical_columns.append(col)
                    
        return categorical_columns
    
    def _manage_high_cardinality(self, series):
        """
        Handle high-cardinality categorical features by limiting to most frequent values.
        
        Parameters:
        -----------
        series : pandas.Series
            Categorical data to process
            
        Returns:
        --------
        pandas.Series : Processed series with reduced cardinality
        """
        value_counts = series.value_counts()
        if len(value_counts) <= self.max_categories:
            return series
            
        # Keep only the top N categories
        top_categories = value_counts.nlargest(self.max_categories).index.tolist()
        return series.apply(lambda x: x if x in top_categories else 'Other')
    
    def _create_numerical_transformers(self):
        """
        Create transformer pipelines for numerical columns based on distribution analysis.
        
        Returns:
        --------
        list : List of (column_name, transformer) tuples for each numerical column
        """
        transformers = []
        
        for col in self.numerical_columns:
            scaling_method = self.scaling_decisions.get(col, 'standard')
            
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            else:  # min_max
                scaler = MinMaxScaler()
                
            transformers.append((col, scaler, [col]))
            
        return transformers
    
    def fit(self, X, y=None):
        """
        Analyze dataset and determine appropriate preprocessing strategies.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input dataset
        y : array-like, optional
            Target variable (not used in preprocessing, but kept for pipeline compatibility)
            
        Returns:
        --------
        self : Returns self
        """
        # Make a copy to avoid modifying the original
        df = X.copy()
        
        # Detect categorical columns
        self.categorical_columns = self._detect_categorical_columns(df)
        self.numerical_columns = [col for col in df.columns if col not in self.categorical_columns]
        
        # Analyze numerical distributions
        for col in self.numerical_columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                stats = self._analyze_distribution(df[col])
                self.column_stats[col] = stats
                self.scaling_decisions[col] = stats['scaling_method']
        
        # Create preprocessing pipelines
        numerical_transformers = self._create_numerical_transformers()
        
        # Handle categorical columns with one-hot encoding
        for col in self.categorical_columns:
            # Apply cardinality management
            if df[col].nunique() > self.cardinality_threshold:
                df[col] = self._manage_high_cardinality(df[col])
                self.categorical_mappings[col] = df[col].value_counts().index.tolist()
                
        # Create the final preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', ColumnTransformer(numerical_transformers), self.numerical_columns),
                ('cat', 'passthrough', self.categorical_columns)
            ]
        )
        
        # Fit the preprocessor
        if len(self.numerical_columns) > 0 or len(self.categorical_columns) > 0:
            self.preprocessor.fit(df)
            
        return self
    
    def transform(self, X):
        """
        Apply preprocessing transformations to the dataset.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input dataset to transform
            
        Returns:
        --------
        pandas.DataFrame : Transformed dataset
        """
        # Make a copy to avoid modifying the original
        df = X.copy()
        
        # Apply categorical cardinality management
        for col in self.categorical_columns:
            if col in self.categorical_mappings:
                # Apply the same mapping as during fit
                df[col] = df[col].apply(
                    lambda x: x if x in self.categorical_mappings[col] else 'Other'
                )
        
        # Apply preprocessing
        if len(self.numerical_columns) > 0 or len(self.categorical_columns) > 0:
            transformed_array = self.preprocessor.transform(df)
            
            # Convert back to DataFrame with appropriate column names
            # First, get the numerical columns (already transformed)
            result_cols = []
            for col in self.numerical_columns:
                result_cols.append(col)
            
            # Then, add categorical columns
            for col in self.categorical_columns:
                result_cols.append(col)
            
            return pd.DataFrame(transformed_array, columns=result_cols, index=X.index)
        else:
            return df
        
    def fit_transform(self, X, y=None):
        """
        Fit the preprocessor to the data and then transform it.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input dataset
        y : array-like, optional
            Target variable (not used in preprocessing, but kept for pipeline compatibility)
            
        Returns:
        --------
        pandas.DataFrame : Transformed dataset
        """
        return self.fit(X, y).transform(X)
    
    def get_preprocessing_report(self):
        """
        Generate a detailed report of the preprocessing decisions.
        
        Returns:
        --------
        dict : Report of preprocessing decisions and statistics
        """
        report = {
            'numerical_features': len(self.numerical_columns),
            'categorical_features': len(self.categorical_columns),
            'scaling_methods': {
                'standard': sum(1 for v in self.scaling_decisions.values() if v == 'standard'),
                'min_max': sum(1 for v in self.scaling_decisions.values() if v == 'min_max'),
                'robust': sum(1 for v in self.scaling_decisions.values() if v == 'robust'),
            },
            'high_cardinality_features': len(self.categorical_mappings),
            'feature_details': {
                col: {
                    'type': 'numerical' if col in self.numerical_columns else 'categorical',
                    'scaling': self.scaling_decisions.get(col, None) if col in self.numerical_columns else None,
                    'cardinality': len(self.categorical_mappings.get(col, [])) if col in self.categorical_mappings else None,
                    **self.column_stats.get(col, {})
                } for col in self.numerical_columns + self.categorical_columns
            }
        }
        return report
    
    def visualize_distributions(self, X, figsize=(15, 10)):
        """
        Visualize the distributions of numerical features before and after transformation.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input dataset
        figsize : tuple
            Figure size for the plots
        """
        if len(self.numerical_columns) == 0:
            print("No numerical columns to visualize")
            return
            
        X_transformed = self.transform(X)
        
        fig, axes = plt.subplots(len(self.numerical_columns), 2, figsize=figsize)
        if len(self.numerical_columns) == 1:
            axes = axes.reshape(1, -1)
            
        for i, col in enumerate(self.numerical_columns):
            # Original distribution
            sns.histplot(X[col].dropna(), kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'Original: {col}')
            
            # Transformed distribution
            sns.histplot(X_transformed[col].dropna(), kde=True, ax=axes[i, 1])
            axes[i, 1].set_title(f'Transformed: {col} ({self.scaling_decisions.get(col, "unknown")})')
            
        plt.tight_layout()
        return fig


# Example usage within an AutoML pipeline
class CategoricalEncoder:
    """
    One-hot encoding implementation with automatic handling of high-cardinality features.
    """
    
    def __init__(self, max_categories=20, drop_first=True):
        """
        Initialize the categorical encoder.
        
        Parameters:
        -----------
        max_categories : int
            Maximum number of categories to keep for high-cardinality features
        drop_first : bool
            Whether to drop the first category level (to prevent dummy variable trap)
        """
        self.max_categories = max_categories
        self.drop_first = drop_first
        self.encoders = {}
        self.category_maps = {}
        
    def fit(self, X, y=None):
        """
        Fit the encoder on categorical columns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input dataset with categorical columns
        y : array-like, optional
            Target variable (not used, kept for compatibility)
            
        Returns:
        --------
        self : Returns self
        """
        for col in X.columns:
            # Get value counts and determine most frequent categories
            value_counts = X[col].value_counts()
            
            if len(value_counts) > self.max_categories:
                # Keep only the top N categories
                top_categories = value_counts.nlargest(self.max_categories).index.tolist()
                self.category_maps[col] = top_categories
                
        return self
    
    def transform(self, X):
        """
        Transform categorical columns to one-hot encoded columns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input dataset with categorical columns
            
        Returns:
        --------
        pandas.DataFrame : One-hot encoded dataset
        """
        df = X.copy()
        
        # Apply category mapping for high-cardinality features
        for col in df.columns:
            if col in self.category_maps:
                df[col] = df[col].apply(lambda x: x if x in self.category_maps[col] else 'Other')
                
        # Apply one-hot encoding
        df_encoded = pd.get_dummies(df, drop_first=self.drop_first)
        
        return df_encoded
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input dataset with categorical columns
        y : array-like, optional
            Target variable (not used, kept for compatibility)
            
        Returns:
        --------
        pandas.DataFrame : One-hot encoded dataset
        """
        return self.fit(X, y).transform(X)


# Sample function to run the entire preprocessing pipeline
def preprocess_dataset(df, target_column=None, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline for an AutoML system.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    target_column : str, optional
        Name of the target column
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Preprocessed data and preprocessing information
    """
    # Separate features and target
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None
    
    # Initialize preprocessors
    feature_preprocessor = AutoMLPreprocessor(
        skew_threshold=0.5,
        kurt_threshold=1.0,
        outlier_threshold=0.05,
        cardinality_threshold=10,
        max_categories=20
    )
    
    # Analyze and transform numerical features
    X_preprocessed = feature_preprocessor.fit_transform(X)
    
    # Generate preprocessing report
    preprocessing_report = feature_preprocessor.get_preprocessing_report()
    
    # One-hot encode categorical features
    categorical_encoder = CategoricalEncoder(max_categories=20, drop_first=True)
    categorical_columns = feature_preprocessor.categorical_columns
    
    if categorical_columns:
        X_cat = X_preprocessed[categorical_columns]
        X_num = X_preprocessed.drop(columns=categorical_columns)
        
        X_cat_encoded = categorical_encoder.fit_transform(X_cat)
        
        # Combine numerical and encoded categorical features
        X_final = pd.concat([X_num, X_cat_encoded], axis=1)
    else:
        X_final = X_preprocessed
    
    # Split into train and test sets if target is available
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=test_size, random_state=random_state
        )
        
        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessing_report': preprocessing_report,
            'feature_preprocessor': feature_preprocessor,
            'categorical_encoder': categorical_encoder
        }
    else:
        result = {
            'X_processed': X_final,
            'preprocessing_report': preprocessing_report,
            'feature_preprocessor': feature_preprocessor,
            'categorical_encoder': categorical_encoder
        }
    
    return result


# Example usage demonstration
if __name__ == "__main__":
    # Create a sample dataset
    np.random.seed(42)
    
    # Sample data with mixed distributions and types
    data = {
        'normal_feature': np.random.normal(0, 1, 1000),
        'skewed_feature': np.random.exponential(2, 1000),
        'outlier_feature': np.concatenate([np.random.normal(0, 1, 950), np.random.normal(10, 1, 50)]),
        'categorical_low': np.random.choice(['A', 'B', 'C'], 1000),
        'categorical_high': np.random.choice([f'Category_{i}' for i in range(30)], 1000, 
                                             p=[0.1, 0.1, 0.1, 0.1, 0.1] + [0.5/25] * 25),
        'binary_target': np.random.choice([0, 1], 1000)
    }
    
    df = pd.DataFrame(data)
    
    # Run preprocessing
    result = preprocess_dataset(df, target_column='binary_target')
    
    # Display preprocessing report
    print("Preprocessing Report:")
    for key, value in result['preprocessing_report'].items():
        if key != 'feature_details':
            print(f"{key}: {value}")
    
    # Show feature scaling decisions
    print("\nFeature Scaling Decisions:")
    for col, details in result['preprocessing_report']['feature_details'].items():
        if details['type'] == 'numerical':
            print(f"{col}: {details['scaling']} (skewness: {details.get('skewness', 'N/A'):.2f}, "
                  f"outlier ratio: {details.get('outlier_ratio', 'N/A'):.2f})")
    
    # Show categorical encoding information
    print("\nCategorical Feature Information:")
    for col, details in result['preprocessing_report']['feature_details'].items():
        if details['type'] == 'categorical':
            print(f"{col}: cardinality={details['cardinality'] or 'N/A'}")
    
    # Show transformed data shape
    print(f"\nTransformed Train Data Shape: {result['X_train'].shape}")
    print(f"Transformed Test Data Shape: {result['X_test'].shape}")
    
    # Optional: Visualize distributions
    result['feature_preprocessor'].visualize_distributions(df.drop(columns=['binary_target']))
    plt.show()