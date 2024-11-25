import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tpot import TPOTClassifier, TPOTRegressor
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.express as px
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import sweetviz as sv

# Updated function to preprocess the data, including scaling and handling categorical features
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder





def preprocess_data(df, target_column, scaler_type='standard', train_size=0.8):
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if there are only numerical or categorical columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Split data into training and testing sets based on user-selected train_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=42)

    if len(numerical_cols) > 0:
        # Impute missing values for numerical columns (mean imputation)
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        # Scale the numerical features based on scaler_type
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler type. Choose 'standard' or 'minmax'.")

        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    if len(categorical_cols) > 0:
        # Impute missing values for categorical columns (mode imputation)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        # One-hot encode categorical features
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])

        # Convert encoded arrays into DataFrame
        X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_cols))

        # Concatenate the encoded columns back with the rest of the data
        X_train = pd.concat([X_train.drop(columns=categorical_cols).reset_index(drop=True), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols).reset_index(drop=True), X_test_encoded], axis=1)

    return X_train, X_test, y_train, y_test


# Check if dataset exists
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    # st.info("This project application helps you build and explore your data.")

# Upload dataset
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)


# Exploratory Data Analysis (Profiling)
if choice == "Profiling":
    st.title("Dataset Insights")
    profile = ProfileReport(df)  # Create the profile report
    st_profile_report(profile)  # Display the profile report in Streamlit




# Modeling
if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    
    # Determine task type
    task_type = 'regression' if df[chosen_target].dtype in ['int64', 'float64'] else 'classification'
    
    # Hyperparameter customization
    st.sidebar.header("Model Hyperparameters")
    generations = st.sidebar.number_input("Iterations", min_value=1, max_value=100, value=5)
    population_size = st.sidebar.number_input("Total models", min_value=1, max_value=100, value=20)
    # verbosity = st.sidebar.selectbox("Metrics", [0, 1, 2, 3], index=2)
    
    if st.button('Run Modelling'):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Preprocess the data and split it into train/test
        X_train, X_test, y_train, y_test = preprocess_data(df, chosen_target)
        
        # Update progress bar
        progress_bar.progress(25)
        status_text.text("Preprocessing Complete. Training model...")
        
        # Initialize the TPOT model based on task type
        if task_type == 'classification':
            model = TPOTClassifier(verbosity=2, generations=generations, population_size=population_size, random_state=42)
        else:
            model = TPOTRegressor(verbosity=2, generations=generations, population_size=population_size, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Update progress bar
        progress_bar.progress(75)
        status_text.text("Model training complete. Evaluating the model...")

        # Check if models have been evaluated
        evaluated_individuals = model.evaluated_individuals_
        if not evaluated_individuals:
            st.error("No models were evaluated. Please check the dataset or the configuration.")
            st.write("Possible issues could be invalid data or insufficient iterations.")
        else:
            # Debugging: Inspect the first evaluated individual
            # first_individual = next(iter(evaluated_individuals.values()))
            # st.write(f"Inspecting first evaluated individual: {first_individual}")
            
            # Directly access the best model (highest score)
            best_pipeline = max(evaluated_individuals, key=lambda x: evaluated_individuals[x].get('internal_cv_score', 0))

            best_model = evaluated_individuals[best_pipeline]

            # Show metrics for the best model
            st.write(f"Best Model: {best_pipeline}")
            st.write(f"Best Model Score (internal CV score): {best_model.get('internal_cv_score', 'N/A'):.4f}")

            # Evaluate the best model on the test data
            if task_type == 'classification':
                # For classification, use the best pipeline
                y_pred = model.fitted_pipeline_.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy of the best model: {accuracy:.4f}")
                
                # Plot Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)

                # Map the 'Yes'/'No' values to 1/0 for y_test
                y_test_binary = y_test.map({'No': 0, 'Yes': 1})
                
                # For ROC Curve
                fpr, tpr, _ = roc_curve(y_test_binary, model.fitted_pipeline_.predict_proba(X_test)[:, 1])
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc='lower right')
                st.pyplot(fig)
            
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test_binary, model.fitted_pipeline_.predict_proba(X_test)[:, 1])
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(recall, precision, color='blue', lw=2)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                st.pyplot(fig)

            else:
                # For regression
                y_pred = model.fitted_pipeline_.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                st.write(f"R^2 Score of the best model: {r2:.4f}")

        evaluated_individuals = model.evaluated_individuals_
        
        if not evaluated_individuals:
            st.error("No models were evaluated. Please check the dataset or the configuration.")
            st.write("Possible issues could be invalid data or insufficient iterations.")
        else:
            # Debugging: Inspect the first evaluated individual
            first_individual = next(iter(evaluated_individuals.values()))
            st.write(f"Inspecting first evaluated individual: {first_individual}")
            
            # Directly access the best model (highest score)
            best_pipeline = max(evaluated_individuals, key=lambda x: evaluated_individuals[x].get('internal_cv_score', 0))

            best_model = evaluated_individuals[best_pipeline]

            # Show metrics for the best model
            st.write(f"Best Model: {best_pipeline}")
            st.write(f"Best Model Score (internal CV score): {best_model.get('internal_cv_score', 'N/A'):.4f}")

            # Save the best model pipeline to a .zip file
            model.export('best_model.zip')
            st.success("Best model saved as 'best_model.zip'")

        # Final update to progress bar
        progress_bar.progress(100)
        status_text.text("Task completed successfully!")

# Downloading Model
if choice == "Download":
    st.title("Download Model")
    if os.path.exists('./best_model.zip'):
        st.download_button("Download Trained Model", './best_model.zip')
    else:
        st.error("Model not trained yet. Please train it first.")
