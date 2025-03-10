import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tpot import TPOTClassifier, TPOTRegressor
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, mean_absolute_error, mean_squared_error
import plotly.express as px
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import warnings
import re
from collections import Counter
import spacy
import nltk
from nltk.corpus import wordnet
from fuzzywuzzy import process

warnings.filterwarnings("ignore")

# Load NLP model
nlp = spacy.load("en_core_web_md")

# Ensure WordNet is downloaded only once
nltk.download("wordnet", quiet=True)

# Set similarity and fuzzy matching thresholds
SIMILARITY_THRESHOLD = 0.6
FUZZY_MATCH_THRESHOLD = 75

def get_synonyms(word):
    """Get synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))  # Convert underscores to spaces
    return list(synonyms)

def get_vector(text):
    """Generate a word vector using SpaCy."""
    doc = nlp(text)
    if doc.vector_norm == 0:  # Avoid division by zero issues
        return np.zeros((nlp.vocab.vectors_length,))
    return doc.vector

def detect_target_column(user_prompt, df):
    """Detect all relevant columns based on regex, vector similarity, and fuzzy matching."""
    
    user_prompt_clean = re.sub(r"[^\w\s]", "", user_prompt.lower())  # Remove special characters
    prompt_words = user_prompt_clean.split()  # Convert to a list of words
    
    matched_columns = set()  # Use a set to avoid duplicates

    # --- CODE 1: Regex-Based Matching ---
    for col in df.columns:
        col_clean = re.sub(r"[\W_]+", " ", col.lower()).strip()  # Normalize column names
        col_words = col_clean.split()  # Convert column name to a list of words
        
        # Check if all words in col_words exist in prompt_words (ignoring order & pluralization)
        if all(any(word.rstrip("s") == prompt_word.rstrip("s") for prompt_word in prompt_words) for word in col_words):
            matched_columns.add(col)

    if matched_columns:
        return list(matched_columns)
    
    # --- CODE 2: Vector-Based Matching ---
    user_vector = get_vector(user_prompt_clean)  # Convert prompt to vector
    
    for col in df.columns:
        col_name = col.lower().replace("_", " ")  # Normalize column name
        col_vector = get_vector(col_name)  # Convert column to vector

        # Compute cosine similarity
        norm_user_vector = np.linalg.norm(user_vector)
        norm_col_vector = np.linalg.norm(col_vector)

        if norm_user_vector == 0 or norm_col_vector == 0:
            continue  # Skip this column if vector is empty

        similarity = np.dot(user_vector, col_vector) / (norm_user_vector * norm_col_vector)

        if similarity >= SIMILARITY_THRESHOLD:
            matched_columns.add(col)

    if matched_columns:
        return list(matched_columns)
    
    # --- CODE 3: Synonym-Based Matching ---
    for col in df.columns:
        col_name = col.lower().replace("_", " ")
        col_words = col_name.split()  # Get individual words

        for word in col_words:
            synonyms = get_synonyms(word)  # Get synonyms dynamically
            if any(synonym in user_prompt_clean for synonym in synonyms):
                matched_columns.add(col)

    # --- CODE 4: Fuzzy Matching ---
    fuzzy_matches = process.extract(user_prompt_clean, df.columns, limit=5)  # Get top 5 fuzzy matches
    for match, score in fuzzy_matches:
        if score >= FUZZY_MATCH_THRESHOLD:
            matched_columns.add(match)

    return list(matched_columns)  # Convert set to list before returning



# Function to determine task type
def detect_task_type(df, target_column):
    """Determines whether the dataset is for classification or regression."""
    unique_values = df[target_column].nunique()
    
    if df[target_column].dtype in ['object', 'category']:  
        return 'classification'  # Categorical target -> classification
    
    if unique_values <= 20:  
        return 'classification'  # Limited unique values (discrete labels) -> classification
    
    if unique_values > 20:  
        return 'regression'  # Many unique values (continuous output) -> regression

    return 'classification'  # Default case

# Function to preprocess the dataset
def preprocess_data(df, target_column, scaler_type='standard', train_size=0.8):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=42)

    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler type. Choose 'standard' or 'minmax'.")

        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        encoder = OneHotEncoder(sparse_output=False, drop='first')
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])

        X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_cols))

        X_train = pd.concat([X_train.drop(columns=categorical_cols).reset_index(drop=True), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols).reset_index(drop=True), X_test_encoded], axis=1)

    return X_train, X_test, y_train, y_test

# Check if dataset exists
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

# Ensure session state initialization
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Upload"
if "df" not in st.session_state:
    st.session_state.df = None

# Sidebar navigation
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Define Task", "Modelling", "Download"], index=["Upload", "Profiling", "Define Task", "Modelling", "Download"].index(st.session_state.selected_tab))

# Upload dataset
if choice == "Upload":
    st.session_state.selected_tab = "Upload"
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=["csv"])
    
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

# Profiling (Exploratory Data Analysis)
elif choice == "Profiling":
    st.session_state.selected_tab = "Profiling"
    st.title("Dataset Insights")

    if df is not None:
        profile = ProfileReport(df)
        st_profile_report(profile)
    else:
        st.error("❌ Please upload a dataset first!")

# Define Task
elif choice == "Define Task":
    st.session_state.selected_tab = "Define Task"
    st.title("Dataset Columns Overview")

    if df is not None:
        st.write("Here are the available columns in your dataset:")
        st.write(df.columns.tolist())

        user_prompt = st.text_area("Describe what you want to predict:", key="user_prompt")
        
        # Reset chosen_target when a new user prompt is given
        if user_prompt and "chosen_target" in st.session_state:
            del st.session_state.chosen_target

        if st.button('Run Modelling'):
            st.session_state.selected_tab = "Modelling"
            st.rerun()  # Forces Streamlit to refresh the page
    else:
        st.error("❌ Please upload a dataset first!")

# Modelling
elif choice == "Modelling":
    st.session_state.selected_tab = "Modelling"
    st.title("Define Your Prediction Task")

    if df is None:
        st.error("❌ Please upload a dataset first!")
    else:
        if 'chosen_target' not in st.session_state:
            print("inininin")
            user_prompt = st.session_state.get('user_prompt', "")
            detected_targets = detect_target_column(user_prompt, df)

            if isinstance(detected_targets, list):
                if len(detected_targets) > 1:
                    chosen_target = st.selectbox('Multiple possible target columns detected. Please select one:', detected_targets)
                elif len(detected_targets) == 1:
                    chosen_target = detected_targets[0]
                    st.success(f"Automatically selected target column: {chosen_target}")
                else:
                    chosen_target = st.selectbox('No clear match found. Choose the Target Column manually:', df.columns)
            else:
                chosen_target = detected_targets
                st.success(f"Automatically selected target column: {chosen_target}")

            st.session_state.chosen_target = chosen_target  # Store in session state
        else:
            chosen_target = st.session_state.chosen_target  # Use stored value
            st.success(f"Using previously selected target column: {chosen_target}")

        # Detect the task type
        task_type = detect_task_type(df, chosen_target)
        st.write(f"Detected Task Type: **{task_type.capitalize()}**")

        # Sidebar for hyperparameters
        st.sidebar.header("Model Hyperparameters")
        generations = st.sidebar.number_input("Iterations", min_value=1, max_value=100, value=1)
        population_size = st.sidebar.number_input("Total models", min_value=1, max_value=100, value=5)

        if st.button('Run Modelling'):
            progress_bar = st.progress(0)
            status_text = st.empty()

            X_train, X_test, y_train, y_test = preprocess_data(df, chosen_target)
            progress_bar.progress(25)
            status_text.text("Preprocessing Complete. Training model...")

            if task_type == 'classification':
                model = TPOTClassifier(verbosity=2, generations=generations, population_size=population_size, random_state=42)
            else:
                model = TPOTRegressor(verbosity=2, generations=generations, population_size=population_size, random_state=42)

            model.fit(X_train, y_train)
            progress_bar.progress(75)
            status_text.text("Model training complete. Evaluating the model...")

            best_pipeline = model.fitted_pipeline_
            st.write(f"Best Model: {best_pipeline}")

            if task_type == 'classification':
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.4f}")

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

            else:
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                st.write(f"R² Score: {r2:.4f}")
                st.write(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

            model.export('best_model.zip')
            st.success("Best model saved as 'best_model.zip'")
            progress_bar.progress(100)
            status_text.text("Task completed successfully!")

# Download Model
elif choice == "Download":
    st.session_state.selected_tab = "Download"
    st.title("Download Model")

    if os.path.exists('./best_model.zip'):
        st.download_button("Download Trained Model", './best_model.zip')
    else:
        st.error("❌ Model not trained yet. Please train it first!")