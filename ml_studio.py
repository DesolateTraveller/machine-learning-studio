import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="Classification Model Comparison", layout="wide")

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

def main():
    st.title("📊 Classification Model Comparison App")

    st.sidebar.title("Configuration")
    st.sidebar.info('Upload your dataset and configure the classification task.', icon="ℹ️")

    # Upload dataset
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        # Read dataset
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("Dataset successfully loaded!")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Select target variable
        target_variable = st.sidebar.selectbox("Select Target Variable", options=df.columns)
        if target_variable:
            X = df.drop(columns=[target_variable])
            y = df[target_variable]

            # Check if classification type is binary or multiclass
            unique_classes = y.nunique()
            if unique_classes == 2:
                classification_type = 'Binary'
            else:
                classification_type = 'Multi-Class'

            st.sidebar.markdown(f"**Detected Problem Type:** {classification_type}")

            # Split dataset
            test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=30, step=5)
            random_state = st.sidebar.number_input("Random State", value=42, step=1)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state, stratify=y
            )

            # Define classifiers
            classifiers = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Ridge Classifier": RidgeClassifier(),
                "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                "Random Forest": RandomForestClassifier(),
                "Extra Trees": ExtraTreesClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "LightGBM": LGBMClassifier(),
                "CatBoost": CatBoostClassifier(verbose=0),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(),
                "SVM (Linear Kernel)": SVC(kernel='linear', probability=True),
                "Dummy Classifier": DummyClassifier(strategy="most_frequent")
            }

            # Select classifiers to include
            selected_classifiers = st.sidebar.multiselect(
                "Select Classifiers to Compare",
                options=list(classifiers.keys()),
                default=list(classifiers.keys())
            )

            if selected_classifiers:
                metrics_list = ["Accuracy", "ROC AUC", "Recall", "Precision", "F1 Score", "Cohen's Kappa", "MCC"]
                results = pd.DataFrame(columns=["Model"] + metrics_list)

                progress_bar = st.progress(0)
                for idx, classifier_name in enumerate(selected_classifiers):
                    classifier = classifiers[classifier_name]
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)

                    # Handle probability predictions
                    if hasattr(classifier, "predict_proba"):
                        y_prob = classifier.predict_proba(X_test)
                        if classification_type == 'Binary':
                            y_prob = y_prob[:,1]
                            roc_auc = roc_auc_score(y_test, y_prob)
                        else:
                            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                    else:
                        y_prob = None
                        roc_auc = np.nan

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred, average='weighted')
                    precision = precision_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    kappa = cohen_kappa_score(y_test, y_pred)
                    mcc = matthews_corrcoef(y_test, y_pred)

                    results = results.append({
                        "Model": classifier_name,
                        "Accuracy": accuracy,
                        "ROC AUC": roc_auc,
                        "Recall": recall,
                        "Precision": precision,
                        "F1 Score": f1,
                        "Cohen's Kappa": kappa,
                        "MCC": mcc
                    }, ignore_index=True)

                    progress_bar.progress((idx+1)/len(selected_classifiers))

                # Highlight best models
                best_models = results.loc[results["F1 Score"].idxmax()]

                st.subheader("Model Performance Comparison")
                st.dataframe(
                    results.style.highlight_max(color='lightgreen', subset=metrics_list)
                )

                st.subheader(f"Best Model: {best_models['Model']}")
                st.write(best_models)

                # Visualization options
                st.subheader("Performance Visualization")
                viz_option = st.selectbox(
                    "Select Visualization",
                    options=["Confusion Matrix", "ROC AUC Curve", "Feature Importance"]
                )

                best_classifier = classifiers[best_models['Model']]
                y_pred_best = best_classifier.predict(X_test)

                if viz_option == "Confusion Matrix":
                    cm = confusion_matrix(y_test, y_pred_best)
                    fig, ax = plt.subplots(figsize=(8,6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel("Predicted Labels")
                    ax.set_ylabel("True Labels")
                    ax.set_title(f"Confusion Matrix for {best_models['Model']}")
                    st.pyplot(fig)

                elif viz_option == "ROC AUC Curve":
                    if y_prob is not None:
                        fig, ax = plt.subplots(figsize=(8,6))
                        if classification_type == 'Binary':
                            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                            roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=best_models['Model'])
                            roc_display.plot(ax=ax)
                        else:
                            RocCurveDisplay.from_estimator(best_classifier, X_test, y_test, ax=ax)
                        ax.set_title(f"ROC AUC Curve for {best_models['Model']}")
                        st.pyplot(fig)
                    else:
                        st.warning("ROC AUC curve is not available for this classifier.")

                elif viz_option == "Feature Importance":
                    if hasattr(best_classifier, 'feature_importances_'):
                        importances = best_classifier.feature_importances_
                        feature_names = X.columns
                        feat_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
                        fig, ax = plt.subplots(figsize=(10,6))
                        sns.barplot(x=feat_importance, y=feat_importance.index, ax=ax)
                        ax.set_title(f"Feature Importance for {best_models['Model']}")
                        ax.set_xlabel("Importance Score")
                        ax.set_ylabel("Features")
                        st.pyplot(fig)
                    elif hasattr(best_classifier, 'coef_'):
                        importances = best_classifier.coef_[0]
                        feature_names = X.columns
                        feat_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
                        fig, ax = plt.subplots(figsize=(10,6))
                        sns.barplot(x=feat_importance, y=feat_importance.index, ax=ax)
                        ax.set_title(f"Feature Coefficients for {best_models['Model']}")
                        ax.set_xlabel("Coefficient Value")
                        ax.set_ylabel("Features")
                        st.pyplot(fig)
                    else:
                        st.warning("Feature importance is not available for this classifier.")

    else:
        st.info("Awaiting for dataset to be uploaded.")

if __name__ == "__main__":
    main()
