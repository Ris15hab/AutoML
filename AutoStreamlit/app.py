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
import requests
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore")


COMPARISON_COLUMNS = [
    "num_instances", "num_features", "num_classes", "total_missing_values", 
    "percentage_missing_values", "instances_with_missing_values", 
    "num_numeric_features", "num_symbolic_features", 
    "percentage_numeric_features", "percentage_symbolic_features", 
    "majority_class_percentage", "minority_class_percentage", 
    "majority_class_size", "minority_class_size", 
    "mean", "std_dev", "min", "max", "kurtosis", 
    "skewness", "auto_correlation", "class_entropy"
]


def train_knn_model(meta_dataset_path, comparison_columns):
    meta_dataset = pd.read_csv(meta_dataset_path)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(meta_dataset[comparison_columns])
    knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    knn.fit(scaled_features)
    return meta_dataset, scaler, knn


def calculate_dataset_attributes(dataset):
    attributes = {
        "num_instances": len(dataset),
        "num_features": len(dataset.columns),
        "num_classes": len(dataset.iloc[:, -1].unique()),  # Assuming the last column is the target
        "total_missing_values": dataset.isnull().sum().sum(),
        "percentage_missing_values": dataset.isnull().mean().mean() * 100,
        "instances_with_missing_values": dataset.isnull().any(axis=1).sum(),
        "num_numeric_features": dataset.select_dtypes(include=["number"]).shape[1],
        "num_symbolic_features": dataset.select_dtypes(include=["object", "category"]).shape[1],
        "percentage_numeric_features": dataset.select_dtypes(include=["number"]).shape[1] / len(dataset.columns) * 100,
        "percentage_symbolic_features": dataset.select_dtypes(include=["object", "category"]).shape[1] / len(dataset.columns) * 100,
        "majority_class_percentage": dataset.iloc[:, -1].value_counts(normalize=True).max() * 100,
        "minority_class_percentage": dataset.iloc[:, -1].value_counts(normalize=True).min() * 100,
        "majority_class_size": dataset.iloc[:, -1].value_counts().max(),
        "minority_class_size": dataset.iloc[:, -1].value_counts().min(),
        "mean": dataset.select_dtypes(include=["number"]).mean().mean(),
        "std_dev": dataset.select_dtypes(include=["number"]).std().mean(),
        "min": dataset.select_dtypes(include=["number"]).min().min(),
        "max": dataset.select_dtypes(include=["number"]).max().max(),
        "kurtosis": dataset.select_dtypes(include=["number"]).kurtosis().mean(),
        "skewness": dataset.select_dtypes(include=["number"]).skew().mean(),
        "auto_correlation": dataset.select_dtypes(include=["number"]).autocorr(),  # Approximation
        "class_entropy": -sum(dataset.iloc[:, -1].value_counts(normalize=True) * 
                              np.log2(dataset.iloc[:, -1].value_counts(normalize=True)))  # Entropy
    }
    return pd.DataFrame([attributes])


def find_closest_dataset(uploaded_dataset, meta_dataset, comparison_columns, scaler, knn_model):
    new_dataset_attributes = calculate_dataset_attributes(uploaded_dataset)
    scaled_attributes = scaler.transform(new_dataset_attributes[comparison_columns])
    distances, indices = knn_model.kneighbors(scaled_attributes)
    closest_index = indices[0][0]
    return meta_dataset.iloc[closest_index]


def find_top_models(closest_dataset, meta_dataset, n=3):
    sorted_meta = meta_dataset.sort_values(by="accuracy", ascending=False)
    return sorted_meta.head(n)[["model_id", "model_name", "accuracy"]]

# Function to fetch model hyperparameters from OpenML
def fetch_hyperparameters(model_id):
    url = f"https://www.openml.org/api/v1/json/run/{model_id}"
    response = requests.get(url)
    if response.status_code == 200:
        run_info = response.json()
        if "parameters" in run_info["run"]:
            return {param["name"]: param["value"] for param in run_info["run"]["parameters"]}
    return {}


def execute_models(uploaded_dataset, top_models, target_column):
    X = uploaded_dataset.drop(columns=[target_column])
    y = uploaded_dataset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = []
    
    for _, row in top_models.iterrows():
        model_name = row["model_name"]
        model_id = row["model_id"]
        hyperparameters = fetch_hyperparameters(model_id)
        print(f"Using Model: {model_name} | Model ID: {model_id} | Hyperparameters: {hyperparameters}")
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results.append({"model_name": model_name, "model_id": model_id, "accuracy": accuracy})
    
    return pd.DataFrame(results)

# Main Workflow
def main_workflow(meta_dataset_path, uploaded_file, n=3):
    meta_dataset, scaler, knn_model = train_knn_model(meta_dataset_path, COMPARISON_COLUMNS)
    uploaded_dataset = pd.read_csv(uploaded_file)
    target_column = uploaded_dataset.columns[-1]
    closest_dataset = find_closest_dataset(uploaded_dataset, meta_dataset, COMPARISON_COLUMNS, scaler, knn_model)
    top_models = find_top_models(closest_dataset, meta_dataset, n)
    results = execute_models(uploaded_dataset, top_models, target_column)
    return results

# Example Usage
# meta_dataset_path = "meta_dataset.csv"
# uploaded_file = "new_dataset.csv"
# results = main_workflow(meta_dataset_path, uploaded_file, n=3)
# print("Model Results on Uploaded Dataset:")
# print(results)




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
    

    task_type = 'regression' if df[chosen_target].dtype in ['int64', 'float64'] else 'classification'
    
    # Hyperparameter customization
    st.sidebar.header("Model Hyperparameters")
    generations = st.sidebar.number_input("Iterations", min_value=1, max_value=100, value=5)
    population_size = st.sidebar.number_input("Total models", min_value=1, max_value=100, value=20)
    # verbosity = st.sidebar.selectbox("Metrics", [0, 1, 2, 3], index=2)
    
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


        evaluated_individuals = model.evaluated_individuals_
        if not evaluated_individuals:
            st.error("No models were evaluated. Please check the dataset or the configuration.")
            st.write("Possible issues could be invalid data or insufficient iterations.")
        else:

            # first_individual = next(iter(evaluated_individuals.values()))
            # st.write(f"Inspecting first evaluated individual: {first_individual}")
            

            best_pipeline = max(evaluated_individuals, key=lambda x: evaluated_individuals[x].get('internal_cv_score', 0))

            best_model = evaluated_individuals[best_pipeline]


            st.write(f"Best Model: {best_pipeline}")
            st.write(f"Best Model Score (internal CV score): {best_model.get('internal_cv_score', 'N/A'):.4f}")


            if task_type == 'classification':

                y_pred = model.fitted_pipeline_.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy of the best model: {accuracy:.4f}")
                

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)


                y_test_binary = y_test.map({'No': 0, 'Yes': 1})
                

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
            

                precision, recall, _ = precision_recall_curve(y_test_binary, model.fitted_pipeline_.predict_proba(X_test)[:, 1])
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(recall, precision, color='blue', lw=2)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                st.pyplot(fig)

            else:

                y_pred = model.fitted_pipeline_.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                st.write(f"R^2 Score of the best model: {r2:.4f}")

        evaluated_individuals = model.evaluated_individuals_
        
        if not evaluated_individuals:
            st.error("No models were evaluated. Please check the dataset or the configuration.")
            st.write("Possible issues could be invalid data or insufficient iterations.")
        else:

            first_individual = next(iter(evaluated_individuals.values()))
            st.write(f"Inspecting first evaluated individual: {first_individual}")
            

            best_pipeline = max(evaluated_individuals, key=lambda x: evaluated_individuals[x].get('internal_cv_score', 0))

            best_model = evaluated_individuals[best_pipeline]


            st.write(f"Best Model: {best_pipeline}")
            st.write(f"Best Model Score (internal CV score): {best_model.get('internal_cv_score', 'N/A'):.4f}")

            
            model.export('best_model.zip')
            st.success("Best model saved as 'best_model.zip'")

        
        progress_bar.progress(100)
        status_text.text("Task completed successfully!")

# Downloading Model
if choice == "Download":
    st.title("Download Model")
    if os.path.exists('./best_model.zip'):
        st.download_button("Download Trained Model", './best_model.zip')
    else:
        st.error("Model not trained yet. Please train it first.")
