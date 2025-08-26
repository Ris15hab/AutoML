# AutoML Framework – Meta-Dataset Driven Machine Learning Automation

This project presents a Meta-Learning powered AutoML framework that automates the entire ML pipeline — from dataset ingestion and preprocessing to intelligent model selection, hyperparameter optimization, and deployment.

Unlike conventional AutoML systems that rely heavily on trial-and-error searches, this framework introduces a Meta-Dataset Repository of dataset characteristics and past model performances to guide optimal model recommendation.

<p align="center"><img width="776" height="517" alt="Screenshot 2025-08-25 at 8 15 45 PM" src="https://github.com/user-attachments/assets/16a66c7f-cfca-4c1f-8f8b-53b41584964f" /></p>

## 🚀 Key Features
- Meta-Dataset Repository
  - Curated from OpenML with 187 classification & 94 regression datasets.
  - Captures dataset size, feature types, distributions, class imbalance, and historical model performance
- Automated Data Preprocessing
  - 🧩 Missing Data Handling: Detects MCAR, MAR, MNAR patterns; applies Mean/Median, KNN, or Modified MICE imputation accordingly
  - <p align="center"><img width="776" height="517" alt="Screenshot 2025-08-25 at 8 18 48 PM" src="https://github.com/user-attachments/assets/6d5cec1f-805a-4d9b-b446-304d7a4ff4c6" /></p>
  - 🔤 Semantic Normalization: Resolves spelling variations, abbreviations, and synonyms using NLP & Fuzzy Matching (FuzzyWuzzy + WordNet).
  - 📏 Adaptive Scaling & Normalization: Chooses between Min-Max, Standard, or Robust scaling based on dataset skewness, kurtosis, and outliers.
  - 🔄 Feature Encoding: Efficient one-hot encoding with cardinality management.
- Target Column & Task Identification
  - Identifies the target column using Regex, NLP embeddings, synonym matching, and fuzzy similarity.
  - Dynamically determines whether the task is classification or regression with >96% accuracy
- Meta-Learning Driven Model Selection
  - Uses KNN-based dataset similarity for finding optimal models from the repository.
  - Recommends both model type & tuned hyperparameters based on historical results.
- End-to-End Pipeline Automation
  - Automatically trains, validates, and deploys the best-suited model.
  - Packages trained models into ready-to-use inference scripts.

## 🛠️ Tech Stack
- Frontend: Streamlit (for dataset upload & interactive exploration)
- Backend: Python (Pandas, NumPy, Scikit-Learn)
- Meta-Learning: KNN-based similarity for dataset-to-model mapping
- NLP & Semantic Processing: SpaCy, WordNet, FuzzyWuzzy

## 📸 Demonstrations
<p align="center"><img width="776" height="517"alt="Screenshot 2025-08-25 at 8 27 26 PM" src="https://github.com/user-attachments/assets/cdb362a6-9d02-46b0-9741-7d4439f637a6" /></p>
<p align="center"><img width="776" height="517" alt="Screenshot 2025-08-25 at 8 27 40 PM" src="https://github.com/user-attachments/assets/2dd24046-be6c-4967-b331-ff91d7c331c0" /></p>
<p align="center"><img width="776" height="517" alt="Screenshot 2025-08-25 at 8 28 26 PM" src="https://github.com/user-attachments/assets/bd79d659-4a23-44a4-9bc0-3e914b90b275" /></p>
<p align="center"><img width="1440" height="842" alt="Screenshot 2025-08-25 at 8 30 22 PM" src="https://github.com/user-attachments/assets/9001d111-6196-4f03-bf01-0c23b5fd0a1c" /></p>

## ⚙️ Installation & Setup

Clone the Repository
```bash
git clone https://github.com/Ris15hab/AutoML-MetaDataset.git
cd AutoML-MetaDataset
```

Install Dependencies
```bash
pip install -r requirements.txt
```

Run the Streamlit App
```bash
streamlit run app.py
```

Upload your dataset (CSV format) and follow the interactive workflow.

# 📊 Performance
- Meta-Dataset Scale: 187 classification + 94 regression datasets.
- Imputation Performance:
  - MCAR → Mean/Median (94.2% detection accuracy)
  - MAR → KNN (87.6%)
  - MNAR → Modified MICE (79.3%)
- Target Column Detection: 96.5% accuracy across diverse datasets.
