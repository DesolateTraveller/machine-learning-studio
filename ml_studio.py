import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np

# Function to load dataset
@st.cache
def load_data():
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(file, sep=None, engine='python', encoding='utf-8', parse_dates=True, infer_datetime_format=True)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        df = pd.DataFrame()
    return df
  
# Function to run and evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else np.nan,
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Kappa": cohen_kappa_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

# Define models to evaluate
models = {
    "Logistic Regression": LogisticRegression(),
    "Ridge Classifier": RidgeClassifier(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "CatBoost Classifier": CatBoostClassifier(verbose=0),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Ada Boost Classifier": AdaBoostClassifier(),
    "Extra Trees Classifier": ExtraTreesClassifier(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "Light Gradient Boosting Machine": LGBMClassifier(),
    "K Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Extreme Gradient Boosting": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
    "SVM - Linear Kernel": SVC(kernel="linear", probability=True)
}

st.title("Classification Algorithm Comparison")

# Load data
file = st.sidebar.file_uploader("**:blue[Choose a file]**",
                                    type=["csv", "xls", "xlsx"], 
                                    accept_multiple_files=False, 
                                    key="file_upload")
if file is not None:
  df = load_data()

    # Display data preview
    st.write("Data Preview")
    #st.dataframe(df.head())

    # User input for target variable
    target = st.selectbox("Select Target Variable", list(df.columns))

    # Split the data
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Evaluate each model and store results
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        metrics["Model"] = name
        results.append(metrics)

    # Create a DataFrame for results
    results_df = pd.DataFrame(results)

    # Highlight the best metrics
    best_metrics = results_df.loc[:, results_df.columns != "Model"].idxmax()

    st.write("Model Performance Comparison")
    st.dataframe(results_df.style.apply(lambda x: ["background-color: lightgreen" if v == best_metrics[c] else "" for v in x], axis=1))

    # Show the best model
    best_model = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
    st.write(f"The best model is: **{best_model}**")

