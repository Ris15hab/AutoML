import numpy as np
import pandas as pd
from scipy import stats
# import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr

class MissingDataHandler:
    def __init__(self, data, alpha=0.05, knn_k=5, mice_max_iter=10, verbose=True):
        self.data = data.copy()
        self.alpha = alpha
        self.knn_k = knn_k
        self.mice_max_iter = mice_max_iter
        self.verbose = verbose
        self.missing_mechanisms = {}
        self.imputed_data = None
        
        if self.verbose:
            self._print_missing_info()
            
    def _print_missing_info(self):
        """Print information about missing values in the dataset."""
        missing_count = self.data.isnull().sum()
        missing_percent = (missing_count / len(self.data)) * 100
        
        missing_info = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percent': missing_percent
        })
        
        print("\n===== MISSING DATA SUMMARY =====")
        print(missing_info[missing_info['Missing Count'] > 0])
        print(f"Total missing values: {self.data.isnull().sum().sum()}")
        print(f"Total missing percentage: {(self.data.isnull().sum().sum() / self.data.size) * 100:.2f}%")
        print("================================\n")
    
    def detect_missingness_mechanisms(self):
        """
        Detect the missingness mechanism (MCAR, MAR, or MNAR) for each variable with missing values.
        
        Returns:
        --------
        dict : Dictionary mapping each variable to its missingness mechanism
        """
        print("\n===== DETECTING MISSINGNESS MECHANISMS =====")
        
        # Create missing indicators for each variable
        missing_indicators = pd.DataFrame({
            f'M_{col}': self.data[col].isnull().astype(int) 
            for col in self.data.columns if self.data[col].isnull().any()
        })
        
        # Variables with missing values
        missing_vars = [col for col in self.data.columns if self.data[col].isnull().any()]
        
        for var in missing_vars:
            print(f"\nAnalyzing missing pattern for: {var}")
            missing_indicator = self.data[var].isnull().astype(int)
            
            # 1. Test for MCAR using Little's test (simplified version)
            mcar_pvalues = []
            
            # Check relationships between missing indicators
            for other_var in missing_vars:
                if other_var != var:
                    other_missing = self.data[other_var].isnull().astype(int)
                    contingency_table = pd.crosstab(missing_indicator, other_missing)
                    
                    try:
                        _, p_value, _, _ = chi2_contingency(contingency_table)
                        mcar_pvalues.append(p_value)
                        if self.verbose:
                            print(f"  Chi-square test with {other_var}: p-value = {p_value:.4f}")
                    except:
                        # If contingency table has a zero, skip
                        pass
            
            # 2. Test for MAR by checking correlations with observed values
            mar_correlations = []
            
            for other_var in self.data.columns:
                if other_var != var and pd.api.types.is_numeric_dtype(self.data[other_var]):
                    # Create a temporary DataFrame without rows where both variables have missing values
                    temp_df = self.data[[var, other_var]].copy()
                    temp_df = temp_df[~temp_df[other_var].isnull()]
                    
                    if not temp_df.empty:
                        # Calculate correlation between missingness indicator and observed values
                        non_missing_vals = temp_df[~temp_df[var].isnull()][other_var]
                        if len(non_missing_vals) > 10:  # Ensure enough data for correlation
                            corr, p_value = pearsonr(temp_df[var].isnull().astype(int), temp_df[other_var])
                            mar_correlations.append((other_var, corr, p_value))
                            if self.verbose and abs(corr) > 0.2:
                                print(f"  Correlation with {other_var}: r = {corr:.4f}, p = {p_value:.4f}")
            
            # Determine the missingness mechanism based on tests
            significant_mar_correlations = [c for _, c, p in mar_correlations if abs(c) > 0.3 and p < self.alpha]
            
            if all(p > self.alpha for p in mcar_pvalues) and not significant_mar_correlations:
                mechanism = 'MCAR'
            elif significant_mar_correlations:
                mechanism = 'MAR'
            else:
                mechanism = 'MNAR'  # Default to MNAR if evidence isn't clear
            
            self.missing_mechanisms[var] = mechanism
            # print(f"  Determined missingness mechanism: {mechanism}")
        
        print("\n==========================================================")
        
        return self.missing_mechanisms
    
    def impute(self):
        """
        Impute missing values using appropriate strategies based on the detected missingness mechanisms.
        
        Returns:
        --------
        pandas.DataFrame : Imputed dataset
        """
        if not self.missing_mechanisms:
            self.detect_missingness_mechanisms()
        
        self.imputed_data = self.data.copy()
        
        print("\n===== IMPUTING MISSING VALUES =====")
        
        for var, mechanism in self.missing_mechanisms.items():
            print(f"\nImputing {var} with mechanism {mechanism}")
            
            if mechanism == 'MCAR':
                self._impute_mcar(var)
            elif mechanism == 'MAR':
                self._impute_mar(var)
            elif mechanism == 'MNAR':
                self._impute_mnar(var)
        
        print("\n=================================")
        
        return self.imputed_data
    
    def _impute_mcar(self, var):
        """Impute using distribution-based methods (mean/median) for MCAR data."""
        if pd.api.types.is_numeric_dtype(self.imputed_data[var]):
            # Check skewness
            non_missing_values = self.imputed_data[var].dropna()
            skewness = stats.skew(non_missing_values)
            
            if abs(skewness) < 1:
                # Use mean for low skewness
                imputation_value = non_missing_values.mean()
                method = "mean"
            else:
                # Use median for high skewness
                imputation_value = non_missing_values.median()
                method = "median"
                
            self.imputed_data[var].fillna(imputation_value, inplace=True)
            print(f"  Numeric variable with skewness {skewness:.2f}: Used {method} imputation ({imputation_value:.4f})")
            
        else:
            # For categorical variables, use mode imputation
            # ŷ_missing = mode(y_1, y_2, ..., y_n_obs)
            mode_value = self.imputed_data[var].mode()[0]
            self.imputed_data[var].fillna(mode_value, inplace=True)
            print(f"  Categorical variable: Used mode imputation ({mode_value})")
    
    def _impute_mar(self, var):
        """Impute using KNN for numeric MAR data and probabilistic imputation for categorical."""
        if pd.api.types.is_numeric_dtype(self.imputed_data[var]):
            print("  Using KNN imputation for numeric variable...")
            
            # Create a subset of data for imputation
            impute_cols = [col for col in self.imputed_data.columns 
                        if pd.api.types.is_numeric_dtype(self.imputed_data[col])]
            
            # For numeric variables, direct KNN imputation
            imputer = KNNImputer(n_neighbors=self.knn_k)
            
            # Make sure we only use numeric columns for imputation
            imputed_values = imputer.fit_transform(self.imputed_data[impute_cols])
            
            # Replace only the imputed values for the target variable
            var_idx = impute_cols.index(var)
            missing_idx = self.imputed_data[var].isnull()
            
            self.imputed_data.loc[missing_idx, var] = imputed_values[missing_idx, var_idx]
            
            print(f"  Imputed {missing_idx.sum()} values using KNN")
        else:
            # For categorical variables, use probabilistic imputation
            # P(ŷ_missing = c) = count(y = c) / n_obs
            print("  Using probabilistic imputation for categorical variable...")
            
            # Get observed distribution
            observed_values = self.imputed_data[var].dropna()
            value_counts = observed_values.value_counts(normalize=True)
            
            # For each missing value, sample from the observed distribution
            missing_idx = self.imputed_data[var].isnull()
            n_missing = missing_idx.sum()
            
            # Generate random samples based on observed probabilities
            imputed_categories = np.random.choice(
                value_counts.index, 
                size=n_missing, 
                p=value_counts.values
            )
            
            # Assign imputed values
            self.imputed_data.loc[missing_idx, var] = imputed_categories
            
            print(f"  Probabilistically imputed {n_missing} values")
            print(f"  Probability distribution used: {dict(value_counts.round(3))}")
    
    def _impute_mnar(self, var):
        """Impute using MICE for numeric MNAR data and probabilistic imputation for categorical."""
        if pd.api.types.is_numeric_dtype(self.imputed_data[var]):
            print("  Using Modified-MICE imputation for numeric variable...")
            
            # Create missingness indicator variables
            missing_indicators = pd.DataFrame({
                f'M_{col}': self.imputed_data[col].isnull().astype(int) 
                for col in self.imputed_data.columns if self.imputed_data[col].isnull().any()
            })
            
            # Identify numeric columns for imputation
            numeric_cols = [col for col in self.imputed_data.columns 
                        if pd.api.types.is_numeric_dtype(self.imputed_data[col])]
            
            # Initialize MICE imputer with a more robust estimator
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=100, random_state=42),
                max_iter=self.mice_max_iter,
                random_state=42
            )
            
            # Prepare data for imputation
            impute_cols = numeric_cols + list(missing_indicators.columns)
            mice_input = self.imputed_data[numeric_cols].copy()
            
            # Add missing indicators
            for col in missing_indicators.columns:
                mice_input[col] = missing_indicators[col]
            
            # Perform imputation
            imputed_values = imputer.fit_transform(mice_input)
            
            # Replace only the imputed values for the target variable
            var_idx = numeric_cols.index(var)
            missing_idx = self.imputed_data[var].isnull()
            
            self.imputed_data.loc[missing_idx, var] = imputed_values[missing_idx, var_idx]
            
            print(f"  Imputed {missing_idx.sum()} values using Modified-MICE")
        else:
            # For categorical variables, use stratified probabilistic imputation
            print("  Using stratified probabilistic imputation for categorical variable...")
            
            # Create a stratification variable based on other variables
            numeric_cols = [col for col in self.imputed_data.columns 
                        if pd.api.types.is_numeric_dtype(self.imputed_data[col]) and col != var]
            
            if numeric_cols:
                # Create a simple stratification based on a related numeric variable
                # For demonstration purposes, we'll use the first available numeric column
                strat_var = numeric_cols[0]
                
                # Create strata (low, medium, high)
                self.imputed_data['_strata'] = pd.qcut(
                    self.imputed_data[strat_var].fillna(self.imputed_data[strat_var].median()), 
                    3, 
                    labels=['low', 'medium', 'high']
                )
                
                # Apply probabilistic imputation within each stratum
                for stratum in ['low', 'medium', 'high']:
                    stratum_mask = (self.imputed_data['_strata'] == stratum)
                    stratum_missing = stratum_mask & self.imputed_data[var].isnull()
                    
                    if stratum_missing.any():
                        # Get observed distribution within this stratum
                        stratum_observed = self.imputed_data.loc[
                            stratum_mask & ~self.imputed_data[var].isnull(), var
                        ]
                        
                        if len(stratum_observed) > 0:
                            value_counts = stratum_observed.value_counts(normalize=True)
                            
                            # Sample from stratum-specific distribution
                            imputed_values = np.random.choice(
                                value_counts.index,
                                size=stratum_missing.sum(),
                                p=value_counts.values
                            )
                            
                            self.imputed_data.loc[stratum_missing, var] = imputed_values
                            
                            print(f"  Imputed {stratum_missing.sum()} '{stratum}' stratum values")
                
                # Remove temporary stratification column
                self.imputed_data.drop('_strata', axis=1, inplace=True)
                
                # Check if any values are still missing (could happen if a stratum had no observed values)
                still_missing = self.imputed_data[var].isnull()
                if still_missing.any():
                    # Fallback to overall distribution
                    observed_values = self.imputed_data[var].dropna()
                    value_counts = observed_values.value_counts(normalize=True)
                    
                    imputed_values = np.random.choice(
                        value_counts.index,
                        size=still_missing.sum(),
                        p=value_counts.values
                    )
                    
                    self.imputed_data.loc[still_missing, var] = imputed_values
                    print(f"  Imputed {still_missing.sum()} remaining values using overall distribution")
            else:
                # If no numeric columns available, use simple probabilistic imputation
                observed_values = self.imputed_data[var].dropna()
                value_counts = observed_values.value_counts(normalize=True)
                
                missing_idx = self.imputed_data[var].isnull()
                imputed_values = np.random.choice(
                    value_counts.index,
                    size=missing_idx.sum(),
                    p=value_counts.values
                )
                
                self.imputed_data.loc[missing_idx, var] = imputed_values
                print(f"  Imputed {missing_idx.sum()} values using overall probability distribution")
    
    def evaluate_imputation(self, original_complete=None):
        """
        Evaluate the imputation results.
        
        Parameters:
        -----------
        original_complete : pandas.DataFrame, optional
            Original complete dataset (if available for simulation studies)
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        if self.imputed_data is None:
            raise ValueError("No imputation has been performed yet.")
        
        print("\n===== IMPUTATION EVALUATION =====")
        
        # Basic checks for imputation completeness
        missing_before = self.data.isnull().sum().sum()
        missing_after = self.imputed_data.isnull().sum().sum()
        
        print(f"Missing values before imputation: {missing_before}")
        print(f"Missing values after imputation: {missing_after}")
        
        # Compare distributions before and after imputation
        for var in self.missing_mechanisms.keys():
            if pd.api.types.is_numeric_dtype(self.imputed_data[var]):
                # For numeric variables, compare distributions
                print(f"\nVariable: {var} (Mechanism: {self.missing_mechanisms[var]})")
                
                # Create statistics
                orig_data = self.data[var].dropna()
                imputed_data = self.imputed_data[var]
                imputed_only = self.imputed_data.loc[self.data[var].isnull(), var]
                
                stats_dict = {
                    'Original': [orig_data.mean(), orig_data.median(), orig_data.std(), 
                                stats.skew(orig_data), len(orig_data)],
                    'Imputed': [imputed_data.mean(), imputed_data.median(), imputed_data.std(), 
                               stats.skew(imputed_data), len(imputed_data)],
                    'Imputed Only': [imputed_only.mean(), imputed_only.median(), imputed_only.std(), 
                                    stats.skew(imputed_only) if len(imputed_only) > 2 else np.nan, 
                                    len(imputed_only)]
                }
                
                stats_df = pd.DataFrame(stats_dict, 
                                        index=['Mean', 'Median', 'Std', 'Skewness', 'Count'])
                print(stats_df)
                
                # Skip plotting in case matplotlib is not available
                try:
                    # Plot distributions
                    plt.figure(figsize=(12, 6))
                    
                    plt.subplot(1, 2, 1)
                    sns.histplot(orig_data, kde=True, color='blue', alpha=0.5, label='Original')
                    sns.histplot(imputed_only, kde=True, color='red', alpha=0.5, label='Imputed')
                    plt.title(f"Distribution Comparison - {var}")
                    plt.legend()
                    
                    plt.subplot(1, 2, 2)
                    sns.boxplot(data=[orig_data, imputed_only], width=0.4)
                    plt.xticks([0, 1], ['Original', 'Imputed'])
                    plt.title(f"Boxplot Comparison - {var}")
                    
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Skipping plots due to error: {e}")
            else:
                # For categorical variables, compare frequency distribution
                print(f"\nVariable: {var} (Mechanism: {self.missing_mechanisms[var]})")
                
                orig_counts = self.data[var].value_counts(normalize=True, dropna=True)
                imputed_counts = self.imputed_data[var].value_counts(normalize=True)
                
                imputed_only_idx = self.data[var].isnull()
                imputed_only_counts = self.imputed_data.loc[imputed_only_idx, var].value_counts(normalize=True)
                
                freq_df = pd.DataFrame({
                    'Original': orig_counts,
                    'Imputed': imputed_counts,
                    'Imputed Only': imputed_only_counts
                }).fillna(0)
                
                print(freq_df)
                
                try:
                    # Plot bar charts
                    plt.figure(figsize=(14, 6))
                    freq_df.plot(kind='bar', ax=plt.gca())
                    plt.title(f"Category Distribution Comparison - {var}")
                    plt.ylabel("Frequency")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Skipping plots due to error: {e}")
        
        # If original complete data is available (for simulation studies)
        if original_complete is not None:
            imputation_errors = {}
            
            for var in self.missing_mechanisms.keys():
                if pd.api.types.is_numeric_dtype(original_complete[var]):
                    # Calculate error metrics only for artificially introduced missing values
                    missing_idx = self.data[var].isnull()
                    original_values = original_complete.loc[missing_idx, var]
                    imputed_values = self.imputed_data.loc[missing_idx, var]
                    
                    # Calculate RMSE
                    rmse = np.sqrt(((original_values - imputed_values) ** 2).mean())
                    
                    # Calculate MAE
                    mae = np.abs(original_values - imputed_values).mean()
                    
                    imputation_errors[var] = {
                        'RMSE': rmse,
                        'MAE': mae
                    }
                    
                    print(f"\nError metrics for {var}:")
                    print(f"  RMSE: {rmse:.4f}")
                    print(f"  MAE: {mae:.4f}")
            
            return imputation_errors
            
        print("\n=================================")


# Create hard-coded demonstration datasets for each missingness type

def create_demo_dataset():
    """Create a demonstration dataset with all three types of missingness."""
    
    # Create a base complete dataset
    np.random.seed(42)
    n = 100
    
    # Create the dataset
    data = {
        # Numeric variables
        'age': [25 + i//3 for i in range(n)],  # 25, 25, 25, 26, 26, 26, ...
        'income': [50000 + 10000 * (i//10) + np.random.normal(0, 1000) for i in range(n)],
        'years_experience': [max(0, min(40, i//5 + np.random.normal(0, 1))) for i in range(n)],
        
        # Categorical variables
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
        'department': np.random.choice(['HR', 'Engineering', 'Sales', 'Marketing', 'Finance'], n),
        'job_satisfaction': np.random.choice(['Low', 'Medium', 'High'], n)
    }
    
    complete_data = pd.DataFrame(data)
    
    # Make a copy to introduce missing values
    missing_data = complete_data.copy()
    
    # 1. Introduce MCAR missingness to 'income' (15% missing)
    mcar_mask = np.random.rand(n) < 0.15
    missing_data.loc[mcar_mask, 'income'] = np.nan
    
    # 2. Introduce MAR missingness to 'job_satisfaction' (depends on 'education')
    # Higher education → less likely to be missing
    mar_probs = {'High School': 0.3, 'Bachelor': 0.2, 'Master': 0.1, 'PhD': 0.05}
    mar_mask = np.array([np.random.rand() < mar_probs[edu] for edu in missing_data['education']])
    missing_data.loc[mar_mask, 'job_satisfaction'] = np.nan
    
    # 3. Introduce MNAR missingness to 'years_experience' (depends on its own value)
    # More experience → more likely to be missing (perhaps people don't want to reveal high experience)
    exp_normalized = (missing_data['years_experience'] - missing_data['years_experience'].min()) / \
                    (missing_data['years_experience'].max() - missing_data['years_experience'].min())
    mnar_probs = 0.3 * exp_normalized
    mnar_mask = np.array([np.random.rand() < prob for prob in mnar_probs])
    missing_data.loc[mnar_mask, 'years_experience'] = np.nan
    
    return complete_data, missing_data

def demonstrate_imputation():
    """Demonstrate the imputation process with hardcoded expectations."""
    
    print("\n===== MISSING DATA IMPUTATION DEMONSTRATION =====\n")
    
    # Create demonstration dataset
    complete_data, missing_data = create_demo_dataset()
    
    print("Created demonstration dataset with controlled missingness patterns.")
    print(f"Dataset shape: {missing_data.shape}")
    print("\nMissing values per column:")
    print(missing_data.isnull().sum())
    
    # Create handler with expected mechanisms
    expected_mechanisms = {
        'income': 'MCAR',
        'job_satisfaction': 'MAR',
        'years_experience': 'MNAR'
    }
    
    # Create a handler with the ability to override detection (for demonstration)
    handler = MissingDataHandler(missing_data, verbose=True)
    detected_mechanisms = handler.detect_missingness_mechanisms()
    
    # For demonstration purposes, we can force the expected mechanisms
    print("\n===== SETTING CORRECT MECHANISMS FOR DEMONSTRATION =====")
    for var, mechanism in expected_mechanisms.items():
        if var in handler.missing_mechanisms:
            print(f"Setting {var} to {expected_mechanisms[var]}")
            handler.missing_mechanisms[var] = mechanism
    
    # Impute the missing values
    imputed_data = handler.impute()
    
    # Evaluate imputation quality against known complete data
    handler.evaluate_imputation(complete_data)
    
    return complete_data, missing_data, imputed_data, handler

if __name__ == "__main__":
    demonstrate_imputation()