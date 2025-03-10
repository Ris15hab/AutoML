import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')


class SemanticMismatchHandler:
    """
    A class to detect and resolve semantic mismatches in categorical data.
    Handles variations, abbreviations, misspellings, and formatting differences.
    """
    
    def __init__(self, similarity_threshold=85, min_freq=0.01, max_categories=100):
        """
        Initialize the semantic mismatch handler.
        
        Parameters:
        ----------
        similarity_threshold : int, default=85
            Threshold for fuzzy string matching (0-100)
        min_freq : float, default=0.01
            Minimum frequency (proportion) required to consider a value as canonical
        max_categories : int, default=100
            Maximum number of unique categories to process (for performance)
        """
        self.similarity_threshold = similarity_threshold
        self.min_freq = min_freq
        self.max_categories = max_categories
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.mappings = {}
        
    def _preprocess_text(self, text):
        """Preprocess text for better matching"""
        if pd.isnull(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove punctuation
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize and lemmatize words, remove stopwords
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        # Remove extra whitespace and join tokens
        text = ' '.join(tokens).strip()
        
        return text
    
    def _select_canonical_value(self, cluster):
        """Select the best canonical value from a cluster"""
        # Use original values (not preprocessed) for the canonical form
        # Choose the most frequent one as canonical
        counts = pd.Series(cluster).value_counts()
        canonical = counts.index[0]
        
        # If there's a tie, prefer the longest one (often more descriptive)
        if len(counts[counts == counts.max()]) > 1:
            longest = max(counts[counts == counts.max()].index, key=len)
            canonical = longest
            
        return canonical
    
    def _get_clusters(self, values, column_name):
        """Group similar values into clusters"""
        # Preprocess all values
        preprocessed_values = {val: self._preprocess_text(val) for val in values}
        
        # Count frequencies
        value_counts = pd.Series(values).value_counts(normalize=True)
        
        # For logging
        print(f"  Processing {len(values)} unique values in column '{column_name}'")
        
        # Start with frequent values as potential cluster centers
        frequent_values = value_counts[value_counts >= self.min_freq].index.tolist()
        
        # If no values meet the frequency threshold, use the top values
        if not frequent_values:
            frequent_values = value_counts.nlargest(min(5, len(value_counts))).index.tolist()
            
        print(f"  Identified {len(frequent_values)} potential cluster centers")
        
        # Build similarity matrix
        similarity_groups = {}
        
        # First pass: group by exact match of preprocessed text
        prep_to_original = {}
        for val in values:
            prep_val = preprocessed_values[val]
            if not prep_val:  # Skip empty strings
                continue
                
            if prep_val not in prep_to_original:
                prep_to_original[prep_val] = []
            prep_to_original[prep_val].append(val)
        
        # Start with these exact match groups
        for prep_val, group in prep_to_original.items():
            canonical = max(group, key=lambda x: value_counts.get(x, 0))
            similarity_groups[canonical] = group
        
        # Second pass: merge similar groups using fuzzy matching
        all_centers = list(similarity_groups.keys())
        merged_groups = {}
        processed_centers = set()
        
        for i, center1 in enumerate(all_centers):
            if center1 in processed_centers:
                continue
                
            prep_center1 = preprocessed_values[center1]
            if not prep_center1:
                continue
                
            # Start a new merged group
            merged_group = similarity_groups[center1].copy()
            merged_centers = [center1]
            processed_centers.add(center1)
            
            # Look for similar centers
            for j in range(i+1, len(all_centers)):
                center2 = all_centers[j]
                if center2 in processed_centers:
                    continue
                    
                prep_center2 = preprocessed_values[center2]
                if not prep_center2:
                    continue
                
                # Calculate similarity
                ratio = fuzz.ratio(prep_center1, prep_center2)
                token_sort = fuzz.token_sort_ratio(prep_center1, prep_center2)
                token_set = fuzz.token_set_ratio(prep_center1, prep_center2)
                
                score = max(ratio, token_sort, token_set)
                
                # If similar enough, merge the groups
                if score >= self.similarity_threshold:
                    merged_group.extend(similarity_groups[center2])
                    merged_centers.append(center2)
                    processed_centers.add(center2)
            
            # Determine the canonical value for the merged group
            if merged_centers:
                best_center = max(merged_centers, key=lambda x: value_counts.get(x, 0))
                merged_groups[best_center] = list(set(merged_group))  # Remove duplicates
        
        # Handle any unprocessed centers
        for center in all_centers:
            if center not in processed_centers:
                merged_groups[center] = similarity_groups[center]
        
        # Now check any values that weren't grouped in the first pass
        unassigned = set(values) - set(val for group in merged_groups.values() for val in group)
        
        for val in unassigned:
            prep_val = preprocessed_values[val]
            if not prep_val:
                continue
                
            # Find best match among group centers
            best_score = 0
            best_center = None
            
            for center in merged_groups.keys():
                prep_center = preprocessed_values[center]
                if not prep_center:
                    continue
                    
                ratio = fuzz.ratio(prep_val, prep_center)
                token_sort = fuzz.token_sort_ratio(prep_val, prep_center)
                token_set = fuzz.token_set_ratio(prep_val, prep_center)
                
                score = max(ratio, token_sort, token_set)
                
                if score > best_score and score >= self.similarity_threshold:
                    best_score = score
                    best_center = center
            
            # If match found, add to that group
            if best_center:
                merged_groups[best_center].append(val)
            else:
                # Create a new singleton group
                merged_groups[val] = [val]
        
        print(f"  Final number of clusters: {len(merged_groups)}")
        return merged_groups
    
    def fit(self, df, columns=None):
        """
        Identify semantic mismatches in specified categorical columns.
        
        Parameters:
        ----------
        df : pandas.DataFrame
            The input dataframe
        columns : list or None, default=None
            List of columns to process. If None, all object/string columns will be processed.
            
        Returns:
        -------
        self : SemanticMismatchHandler
            Returns self for method chaining
        """
        # If columns not specified, use all object columns
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for column in columns:
            if column not in df.columns:
                print(f"Warning: Column '{column}' not found in dataframe.")
                continue
                
            # Skip columns with too many unique values for performance
            unique_values = df[column].dropna().unique()
            if len(unique_values) > self.max_categories:
                print(f"Warning: Column '{column}' has {len(unique_values)} unique values, "
                      f"which exceeds the maximum of {self.max_categories}. Skipping.")
                continue
                
            print(f"Processing column: '{column}'")
            
            # Get clusters of similar values
            clusters = self._get_clusters(unique_values, column)
            
            # Create mapping from all values to their canonical form (cluster center)
            self.mappings[column] = {}
            for center, cluster in clusters.items():
                for val in cluster:
                    self.mappings[column][val] = center
            
            # Report findings
            standardized = sum(1 for val, canon in self.mappings[column].items() if val != canon)
            
            print(f"  Found {standardized} values to standardize out of {len(unique_values)} unique values")
            
            # Show examples of standardizations
            if standardized > 0:
                examples = [(val, canon) for val, canon in self.mappings[column].items() if val != canon]
                sample_size = min(5, len(examples))
                
                print("  Examples of standardizations:")
                for original, canonical in examples[:sample_size]:
                    print(f"    '{original}' → '{canonical}'")
        
        return self
    
    def transform(self, df, columns=None, inplace=False):
        """
        Apply the identified semantic standardizations to the dataframe.
        
        Parameters:
        ----------
        df : pandas.DataFrame
            The input dataframe
        columns : list or None, default=None
            List of columns to transform. If None, all fitted columns will be transformed.
        inplace : bool, default=False
            If True, modifies the dataframe in place. Otherwise, returns a copy.
            
        Returns:
        -------
        pandas.DataFrame
            The transformed dataframe
        """
        if not inplace:
            df = df.copy()
            
        # If columns not specified, use all fitted columns that exist in the dataframe
        if columns is None:
            columns = [col for col in self.mappings.keys() if col in df.columns]
        
        for column in columns:
            if column not in self.mappings:
                print(f"Warning: Column '{column}' has not been fitted.")
                continue
                
            if column not in df.columns:
                print(f"Warning: Column '{column}' not found in dataframe.")
                continue
                
            # Apply mapping
            df[column] = df[column].map(lambda x: self.mappings[column].get(x, x))
            
        return df
    
    def fit_transform(self, df, columns=None, inplace=False):
        """
        Fit and transform in one step.
        """
        self.fit(df, columns)
        return self.transform(df, columns, inplace)
    
    def get_mapping(self, column):
        """
        Get the mapping dictionary for a specific column.
        
        Parameters:
        ----------
        column : str
            The column name
            
        Returns:
        -------
        dict
            A dictionary mapping original values to canonical forms
        """
        if column not in self.mappings:
            print(f"Warning: Column '{column}' has not been fitted.")
            return {}
            
        return {k: v for k, v in self.mappings[column].items() if k != v}
    
    def add_custom_mapping(self, column, mapping_dict):
        """
        Add or update custom mappings for a column.
        
        Parameters:
        ----------
        column : str
            The column name
        mapping_dict : dict
            A dictionary mapping original values to desired canonical forms
            
        Returns:
        -------
        self : SemanticMismatchHandler
            Returns self for method chaining
        """
        if column not in self.mappings:
            self.mappings[column] = {}
            
        self.mappings[column].update(mapping_dict)
        return self


def standardize_values(df, columns=None, similarity_threshold=85, min_freq=0.01, custom_mappings=None):
    """
    Detect and resolve semantic mismatches in categorical data.
    
    Parameters:
    ----------
    df : pandas.DataFrame
        The input dataframe
    columns : list or None, default=None
        List of columns to process. If None, all object/string columns will be processed.
    similarity_threshold : int, default=85
        Threshold for fuzzy string matching (0-100)
    min_freq : float, default=0.01
        Minimum frequency required for a value to be considered canonical
    custom_mappings : dict, default=None
        Dictionary of form {column_name: {original_value: canonical_value, ...}}
        
    Returns:
    -------
    pandas.DataFrame
        Dataframe with standardized categorical values
    SemanticMismatchHandler
        The fitted handler for inspection or further use
    """
    handler = SemanticMismatchHandler(
        similarity_threshold=similarity_threshold,
        min_freq=min_freq
    )
    
    # Apply custom mappings if provided
    if custom_mappings:
        for column, mapping in custom_mappings.items():
            handler.add_custom_mapping(column, mapping)
    
    # Fit and transform
    result_df = handler.fit_transform(df, columns)
    
    return result_df, handler


# Example usage
if __name__ == "__main__":
    # Sample data with semantic mismatches
    data = {
        'city': ['New York', 'NYC', 'newyork', 'New York City', 'NY', 'new york'],
        'state': ['California', 'CA', 'Cali', 'calif', 'California', 'CAL'],
        'category': ['Electronics', 'electronic', 'Electronics', 'Elect.', 'electronics', 'ELECTRONICS']
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print("\n" + "-"*50 + "\n")
    
    # Custom mappings (optional)
    custom_mappings = {
        'state': {'NY': 'New York'}  # Example of a custom mapping
    }
    
    # Standardize the data
    standardized_df, handler = standardize_values(
        df, 
        similarity_threshold=75,
        custom_mappings=custom_mappings
    )
    
    print("\nStandardized DataFrame:")
    print(standardized_df)
    
    # Inspect mappings
    print("\nMappings for 'city' column:")
    city_mappings = handler.get_mapping('city')
    for original, canonical in city_mappings.items():
        print(f"  '{original}' → '{canonical}'")
        
    print("\nMappings for 'state' column:")
    state_mappings = handler.get_mapping('state')
    for original, canonical in state_mappings.items():
        print(f"  '{original}' → '{canonical}'")
        
    print("\nMappings for 'category' column:")
    category_mappings = handler.get_mapping('category')
    for original, canonical in category_mappings.items():
        print(f"  '{original}' → '{canonical}'")