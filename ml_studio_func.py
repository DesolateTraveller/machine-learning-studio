#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st

#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#----------------------------------------
import altair as alt
import plotly.express as px
import plotly.offline as pyoff
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
#----------------------------------------
#from ml_insample import classification_analysis
#----------------------------------------
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

#----------------------------------------
# Model Building
import xgboost as xgb
from sklearn import tree
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
#
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
#import optuna.integration.lightgbm as lgb
from sklearn.metrics import classification_report,confusion_matrix
#----------------------------------------
# Model Performance & Validation
import shap
import scipy.cluster.hierarchy as shc
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, f_regression, chi2, VarianceThreshold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, silhouette_score
from sklearn.metrics import accuracy_score, auc, roc_auc_score, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef, precision_recall_curve
#----------------------------------------
# Model Validation
#----------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso,BayesianRidge, OrthogonalMatchingPursuit, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
#----------------------------------------
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import statsmodels.api as sm
import textwrap
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import ks_2samp, entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

@st.cache_data(ttl="2h")
def load_file(file):
    file_extension = file.name.split('.')[-1].lower()
    try:
        if file_extension == 'csv':
            df = pd.read_csv(file, sep=None, engine='python', encoding='utf-8', parse_dates=True, infer_datetime_format=True)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file)
        elif file_extension in ['parquet', 'pq']:
            df = pd.read_parquet(file)
        else:
            st.error("❌ Unsupported file format. Please upload a CSV, Excel, or Parquet file.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"🚫 Error loading file: {e}")
        return pd.DataFrame()
    
    return df

@st.cache_data(ttl="2h")
def check_missing_values(data):
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    return missing_values 

@st.cache_data(ttl="2h")
def check_outliers(data):
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    outliers = pd.DataFrame(columns=['Column', 'Number of Outliers'])
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        outliers_indices = ((data[column] < Q1 - threshold * IQR) | (data[column] > Q3 + threshold * IQR))
        num_outliers = outliers_indices.sum()
        outliers = outliers._append({'Column': column, 'Number of Outliers': num_outliers}, ignore_index=True)
        return outliers
    
@st.cache_data(ttl="2h")
def handle_numerical_missing_values(data, numerical_strategy):
    imputer = SimpleImputer(strategy=numerical_strategy)
    numerical_features = data.select_dtypes(include=['number']).columns
    data[numerical_features] = imputer.fit_transform(data[numerical_features])
    return data

@st.cache_data(ttl="2h")
def handle_categorical_missing_values(data, categorical_strategy):
    imputer = SimpleImputer(strategy=categorical_strategy, fill_value='no_info')
    categorical_features = data.select_dtypes(exclude=['number']).columns
    data[categorical_features] = imputer.fit_transform(data[categorical_features])
    return data  

@st.cache_data(ttl="2h")
def label_encode(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df

#----------------------------------------
@st.cache_data(ttl="2h")
def onehot_encode(df, column):
    ohe = OneHotEncoder(sparse=False)
    encoded_cols = ohe.fit_transform(df[[column]])
    encoded_df = pd.DataFrame(encoded_cols, columns=[f"{column}_{cat}" for cat in ohe.categories_[0]])
    df = df.drop(column, axis=1).join(encoded_df)
    return df

@st.cache_data(ttl="2h")
def scale_features(df, method):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if method == 'Standard Scaling':
        scaler = StandardScaler()
    elif method == 'Min-Max Scaling':
        scaler = MinMaxScaler()
    elif method == 'Robust Scaling':
        scaler = RobustScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

@st.cache_data(ttl="2h")
def calculate_vif(data):
    X = data.values
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    return vif_data

@st.cache_data(ttl="2h")
def drop_high_vif_variables(data, threshold):
    vif_data = calculate_vif(data)
    high_vif_variables = vif_data[vif_data["VIF"] > threshold]["Variable"].tolist()
    data = data.drop(columns=high_vif_variables)
    return data

@st.cache_data(ttl="2h")
def iterative_vif_filtering(df, threshold):
    df = df.copy()
    iteration = 0
    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        max_vif = vif_data["VIF"].max()
        if max_vif <= threshold:
            st.success(f"✅ All features now have VIF ≤ {threshold}.")
            break
        st.markdown(f"**Iteration {iteration + 1} - VIF values:**")
        st.dataframe(vif_data.sort_values("VIF", ascending=False), hide_index=True)
        drop_feature = vif_data.sort_values("VIF", ascending=False).iloc[0]["feature"]
        st.warning(f"Dropping feature **'{drop_feature}'** with VIF = **{max_vif:.2f}**")
        df.drop(columns=[drop_feature], inplace=True)
        iteration += 1
    return df, vif_data.sort_values("VIF")

#----------------------------------------
def check_linear_regression_assumptions(y_true, y_pred, X, model):
    residuals = y_true - y_pred
    X_const = sm.add_constant(X)

    # 1. Linearity
    linear_corr = np.corrcoef(y_pred, residuals)[0, 1]
    linearity_result = "✅ Linear"
    linearity_remark = "Acceptable" if abs(linear_corr) < 0.3 else "⚠️ Pattern found"
    linearity_thresh = "|Corr(pred, resid)| < 0.3"
    linearity_status = "Accept" if abs(linear_corr) < 0.3 else "Reject"

    # 2. Normality of residuals
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    normality_result = "✅ Normal" if shapiro_p > 0.05 else "⚠️ Not Normal"
    normality_remark = f"p = {shapiro_p:.4f}"
    normality_thresh = "Shapiro-Wilk p > 0.05"
    normality_status = "Accept" if shapiro_p > 0.05 else "Reject"

    # 3. Homoscedasticity (Breusch-Pagan test)
    fig, ax = plt.subplots(figsize=(6,3))
    sns.scatterplot(x=y_pred, y=residuals, ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Residuals", fontsize=8)
    ax.set_title("Residuals vs Predicted Values", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.set_ylim(residuals.min() * 1.2, residuals.max() * 1.2)
    fig.tight_layout()

    bp_stat, bp_pvalue = het_breuschpagan(residuals, X_const)[:2]
    homo_result = "✅ Homoscedastic" if bp_pvalue > 0.05 else "⚠️ Heteroscedastic"
    homo_remark = f"p = {bp_pvalue:.4f} → {'No heteroscedasticity' if bp_pvalue > 0.05 else 'Heteroscedasticity detected'}"
    homo_thresh = "Breusch-Pagan p > 0.05"
    homo_status = "Accept" if bp_pvalue > 0.05 else "Reject"

    # 4. Multicollinearity
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    high_vif = vif_df[vif_df["VIF"] > 5]
    multi_result = "✅ OK" if high_vif.empty else "⚠️ High VIF"
    multi_remark = "No issues" if high_vif.empty else f"Issue(s): {', '.join(high_vif['Feature'])}"
    multi_thresh = "VIF < 5"
    multi_status = "Accept" if high_vif.empty else "Reject"

    # 5. Durbin-Watson
    dw_stat = durbin_watson(residuals)
    dw_result = f"{dw_stat:.4f}"
    dw_remark = "✅ No autocorrelation" if 1.5 <= dw_stat <= 2.5 else "⚠️ Possible autocorrelation"
    dw_thresh = "1.5 < DW < 2.5"
    dw_status = "Accept" if 1.5 <= dw_stat <= 2.5 else "Reject"

    # Final table with status column
    df_result = pd.DataFrame({
        "Check": ["Linearity", "Normality", "Homoscedasticity", "Multicollinearity", "Durbin-Watson"],
        "Result": [linearity_result, normality_result, homo_result, multi_result, dw_remark],
        "Remarks": [linearity_remark, normality_remark, homo_remark, multi_remark, dw_result],
        "Threshold Used": [linearity_thresh, normality_thresh, homo_thresh, multi_thresh, dw_thresh],
        "Status": [linearity_status, normality_status, homo_status, multi_status, dw_status]
    })

    # Display
    col1, col2 = st.columns([1,1])
    with col1:
        st.pyplot(fig, use_container_width=True)
    with col2:
        st.dataframe(df_result, hide_index=True)

    # Overall result
    failed = df_result[df_result["Status"] == "Reject"]
    if failed.empty:
        st.success("✅ All linear regression assumptions are satisfied. Model is acceptable.")
    else:
        st.error(f"❌ Linear regression assumptions not fully met. "
                 f"{len(failed)} issue(s) detected — model reliability may be compromised.")

    return df_result

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred)) if np.all(y_pred > 0) else None
    mape_value = mape(y_true, y_pred)
    return mae, mse, rmse, r2, rmsle, mape_value

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

def evaluate_model_train(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_pred_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "Accuracy": accuracy_score(y_train, y_pred),
        "AUC": roc_auc_score(y_train, y_pred_prob) if y_pred_prob is not None else np.nan,
        "Recall": recall_score(y_train, y_pred),
        "Precision": precision_score(y_train, y_pred),
        "F1 Score": f1_score(y_train, y_pred),
        "Kappa": cohen_kappa_score(y_train, y_pred),
        "MCC": matthews_corrcoef(y_train, y_pred)
    } 
       
#----------------------------------------
def wrap_labels(labels, width=10):
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

def plot_learning_curve(model, X_train, y_train,):
    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    plt.figure(figsize=(8, 3))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-Validation Score")
    plt.title("Learning Curve", fontsize=8)
    plt.xlabel("Training Examples", fontsize=8)
    plt.ylabel("Score (Negative MSE)", fontsize=8)
    plt.legend(loc="best")
    st.pyplot(plt, use_container_width=True)

def plot_validation_curve(model, X_train, y_train, param_name, param_range, ):
    train_scores, val_scores = validation_curve(model, X_train, y_train, param_name=param_name, param_range=param_range, cv=5, scoring='neg_mean_squared_error')
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    plt.figure(figsize=(8, 3))
    plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(param_range, val_scores_mean, 'o-', color="g", label="Cross-Validation Score")
    plt.title("Validation Curve", fontsize=8)
    plt.xlabel(f"Values of {param_name}", fontsize=8)
    plt.ylabel("Score (Negative MSE)", fontsize=8)
    plt.legend(loc="best")
    st.pyplot(plt, use_container_width=True)
    
def plot_lift_curve(y_true, y_proba, ax):
    df = pd.DataFrame({'y': y_true, 'score': y_proba})
    df.sort_values('score', ascending=False, inplace=True)
    df['cumulative_data'] = np.arange(1, len(df) + 1)
    df['cumulative_positives'] = df['y'].cumsum()
    df['lift'] = df['cumulative_positives'] / (df['cumulative_data'] * df['y'].mean())
    ax.plot(df['cumulative_data'] / len(df), df['lift'], color='blue', label='Lift Curve')
    ax.axhline(1, color='gray', linestyle='--', label='Baseline')
    ax.set_title("Lift Curve", fontsize=9)
    ax.set_xlabel("Proportion of Samples", fontsize=8)
    ax.set_ylabel("Lift", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    
def plot_gain_curve(y_true, y_proba, ax):
    df = pd.DataFrame({'y': y_true, 'score': y_proba})
    df.sort_values('score', ascending=False, inplace=True)
    df['cumulative_data'] = np.arange(1, len(df) + 1)
    df['cumulative_positives'] = df['y'].cumsum()
    df['gain'] = df['cumulative_positives'] / df['y'].sum()
    ax.plot(df['cumulative_data'] / len(df), df['gain'], color='green', label='Cumulative Gain')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Baseline')
    ax.set_title("Cumulative Gain Curve", fontsize=9)
    ax.set_xlabel("Proportion of Samples", fontsize=8)
    ax.set_ylabel("Proportion of Positive Responses", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    
#----------------------------------------
def compute_psi(expected, actual, buckets=10):
    """Population Stability Index (PSI)"""
    def scale_range(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-8)

    expected = scale_range(expected)
    actual = scale_range(actual)
    
    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    psi = np.sum((expected_percents - actual_percents) * 
                 np.log((expected_percents + 1e-5) / (actual_percents + 1e-5)))
    return psi

def compute_drift_matrix(train_proba, new_proba):
    train_proba = np.array(train_proba)
    new_proba = np.array(new_proba)

    # Normalize histograms
    def safe_norm(arr):
        hist, _ = np.histogram(arr, bins=100, range=(0,1), density=True)
        hist += 1e-10
        return hist / hist.sum()

    p = safe_norm(train_proba)
    q = safe_norm(new_proba)

    # Metric computations with thresholds and remarks
    results = []

    # 1. Kolmogorov-Smirnov
    ks_p = ks_2samp(train_proba, new_proba).pvalue
    results.append({
        "stat_test": "K-S p_value",
        "drift_score": ks_p,
        "threshold": "p < 0.05",
        "is_drifted": ks_p < 0.05,
        "remarks": "Drift detected" if ks_p < 0.05 else "No significant drift"
    })

    # 2. PSI
    psi = compute_psi(train_proba, new_proba)
    results.append({
        "stat_test": "PSI",
        "drift_score": psi,
        "threshold": "> 0.2",
        "is_drifted": psi > 0.2,
        "remarks": "Drift detected" if psi > 0.2 else "Stable"
    })

    # 3. KL Divergence
    kl = entropy(p, q)
    results.append({
        "stat_test": "Kullback-Leibler divergence",
        "drift_score": kl,
        "threshold": "> 0.5",
        "is_drifted": kl > 0.5,
        "remarks": "High divergence" if kl > 0.5 else "Low divergence"
    })

    # 4. Jensen-Shannon
    js = jensenshannon(p, q)
    results.append({
        "stat_test": "Jensen-Shannon distance",
        "drift_score": js,
        "threshold": "> 0.1",
        "is_drifted": js > 0.1,
        "remarks": "Drift detected" if js > 0.1 else "Acceptable distance"
    })

    # 5. Wasserstein
    wass = wasserstein_distance(train_proba, new_proba)
    results.append({
        "stat_test": "Wasserstein distance (normed)",
        "drift_score": wass,
        "threshold": "> 0.1",
        "is_drifted": wass > 0.1,
        "remarks": "Distribution shifted" if wass > 0.1 else "Stable distribution"
    })

    drift_df = pd.DataFrame(results)
    return drift_df    
    