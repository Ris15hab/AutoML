import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

# Initialize session state
if "target_column" not in st.session_state:
    st.session_state["target_column"] = None
if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = ""
if "df" not in st.session_state:
    st.session_state["df"] = None

# Function to detect target column from user prompt
def detect_target_column(user_prompt, df):
    prompt_lower = user_prompt.lower()
    candidate_columns = [col for col in df.columns if col.lower() in prompt_lower]
    return candidate_columns if candidate_columns else None

# Sidebar Navigation
st.sidebar.title("AutoML App")
choice = st.sidebar.radio("Navigation", ["Define Task", "Modelling"])

# Tab 1: Define Task
if choice == "Define Task":
    st.title("Step 1: Define Your Prediction Task")

    # File upload
    file = st.file_uploader("Upload Your Dataset (CSV)", type=["csv"])
    
    if file:
        df = pd.read_csv(file)
        st.session_state["df"] = df  # Store dataset in session state
        df.to_csv("dataset.csv", index=False)  # Save dataset for later use
        st.write("### Preview of Data:")
        st.dataframe(df)

        # User enters prediction goal
        user_prompt = st.text_area("Describe what you want to predict:", st.session_state["user_prompt"])

        if user_prompt:
            st.session_state["user_prompt"] = user_prompt
            detected_targets = detect_target_column(user_prompt, df)

            if detected_targets and len(detected_targets) == 1:
                st.session_state["target_column"] = detected_targets[0]
                st.success(f"Detected Target Column: **{detected_targets[0]}**")
            elif detected_targets:
                st.session_state["target_column"] = st.selectbox("Select Target Column:", detected_targets)
            else:
                st.session_state["target_column"] = st.selectbox("No match found. Select Target Column:", df.columns)

# Tab 2: Modelling
elif choice == "Modelling":
    st.title("Step 2: Train Your Model")

    # Load dataset
    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv")
        st.session_state["df"] = df
        st.write("### Preview of Data:")
        st.dataframe(df)

        if not st.session_state["target_column"]:
            st.error("Please define your task in the 'Define Task' tab first.")
        else:
            target = st.session_state["target_column"]
            st.success(f"Using Target Column: **{target}**")

            # Select features and target
            X = df.drop(columns=[target])
            y = df[target]

            # Check if the target is categorical or numerical
            is_classification = y.dtype == "object" or y.nunique() < 10  # Classification if few unique values

            # Convert categorical target to numeric
            if is_classification:
                y = y.astype("category").cat.codes

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Selection
            model = RandomForestClassifier() if is_classification else RandomForestRegressor()

            # Train Model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluation
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"### Model Accuracy: {accuracy:.2f}")
            else:
                mae = mean_absolute_error(y_test, y_pred)
                st.write(f"### Mean Absolute Error: {mae:.2f}")

            st.success("Model Training Completed ✅")

    else:
        st.error("No dataset found. Please upload in the 'Define Task' tab.")
