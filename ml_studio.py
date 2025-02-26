#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Template Graphics
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit.components.v1 as components
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
#from streamlit_extras.stoggle import stoggle
#from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
#----------------------------------------
import os
import time
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from random import randint
#----------------------------------------
import json
import holidays
import base64
import itertools
import codecs
from datetime import datetime, timedelta, date
#from __future__ import division
#----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#----------------------------------------
#import dabl
import altair as alt
import plotly.express as px
import plotly.offline as pyoff
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
#import scikitplot as skplt
#----------------------------------------
import shutil
import sweetviz as sv
import pygwalker as pyg
from scipy.stats import gaussian_kde
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
from sklearn.model_selection import KFold, cross_val_score, train_test_split
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
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, homogeneity_score, adjusted_rand_score, completeness_score, silhouette_samples
#----------------------------------------
#from pycaret.classification import setup, compare_models, pull, save_model, evaluate_model
#from pycaret.classification import setup, compare_models, predict_model, pull, plot_model, create_model, ensemble_model, blend_models, stack_models, tune_model, save_model
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
st.set_page_config(page_title="ML Studio | v0.3",
                   layout="wide",
                   page_icon="💻",              
                   initial_sidebar_state="auto")
#----------------------------------------
#st.title(f""":rainbow[Machine Learning (ML) Studio]""")
st.markdown(
    """
    <style>
    .title-large {
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title-small {
        text-align: center;
        font-size: 20px;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    <div class="title-large">Machine Learning (ML) Studio</div>
    <div class="title-small">Version : 0.3</div>
    """,
    unsafe_allow_html=True
)
#----------------------------------------
# Set the background image
#st.info('**A lightweight Machine Learning (ML) streamlit app that help to analyse different kind machine learning problems**', icon="ℹ️")
#----------------------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F2F6;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
        z-index: 100;
    }
    .footer p {
        margin: 0;
    }
    .footer .highlight {
        font-weight: bold;
        color: blue;
    }
    </style>

    <div class="footer">
        <p>© 2025 | Created by : <span class="highlight">Avijit Chakraborty</span> | <a href="mailto:avijit.mba18@gmail.com"> 📩 </a></p> <span class="highlight">Thank you for visiting the app | Unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span>
    </div>
    """,
    unsafe_allow_html=True)
#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

st.markdown(
            """
            <style>
                .centered-info {
                display: flex;
                justify-content: center;
                align-items: center;
                font-weight: bold;
                font-size: 15px;
                color: #007BFF; 
                padding: 5px;
                background-color: #E8F4FF; 
                border-radius: 5px;
                border: 1px solid #007BFF;
                margin-top: 5px;
                margin-bottom: 10px;
                }
            </style>
            """,unsafe_allow_html=True,)

@st.cache_data(ttl="2h")
def load_file(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(file, sep=None, engine='python', encoding='utf-8', parse_dates=True, infer_datetime_format=True)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        df = pd.DataFrame()
    return df

#----------------------------------------
@st.cache_data(ttl="2h")
def plot_histograms_with_kde(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the dataset to plot.")
        return
    n_cols = len(numerical_columns) 
    n_rows = (len(numerical_columns) + n_cols - 1) // n_cols  
    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(numerical_columns, 1):
        with st.container():
            fig, ax = plt.subplots(figsize=(25,5)) 
            ax.hist(df[col].dropna(), bins=20, color='skyblue', edgecolor='black', alpha=0.6, density=True)
            kde = gaussian_kde(df[col].dropna())
            x_vals = np.linspace(df[col].min(), df[col].max(), 1000)
            ax.plot(x_vals, kde(x_vals), color='red', lw=2)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            st.pyplot(plt,use_container_width=True)

@st.cache_data(ttl="2h")
def plot_scatter(df, target_column):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the dataset to plot.")
        return
    if target_column not in df.columns:
        st.warning(f"The target column '{target_column}' is not in the dataset.")
        return
    if target_column not in numerical_columns:
        st.warning(f"The target column '{target_column}' is not numerical.")
        return
    numerical_columns = [col for col in numerical_columns if col != target_column]
    for col in numerical_columns:
        with st.container():
            fig, ax = plt.subplots(figsize=(25,5)) 
            ax.scatter(df[col], df[target_column], color='purple', alpha=0.7)
            ax.set_title(f'{target_column} vs {col}')
            ax.set_xlabel(col)
            ax.set_ylabel(target_column)
            ax.grid(True)  
            st.pyplot(plt,use_container_width=True)
#----------------------------------------
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

#----------------------------------------
# Dictionary of metrics
metrics_dict = {
    "Area Under the Curve": 'auc',
    "Discrimination Threshold": 'threshold',
    "Precision-Recall Curve": 'pr',
    "Confusion Matrix": 'confusion_matrix',
    "Class Prediction Error": 'error',
    "Classification Report": 'class_report',
    "Decision Boundary": 'boundary',
    "Recursive Feature Selection": 'rfe',
    "Learning Curve": 'learning',
    "Manifold Learning": 'manifold',
    "Calibration Curve": 'calibration',
    "Validation Curve": 'vc',
    "Dimension Learning": 'dimension',
    "Feature Importance (Top 10)": 'feature',
    "Feature IImportance (all)": 'feature_all',
    "Lift Curve":'lift',
    "Gain Curve": 'gain',
    #"KS Statistic Plot":  'ks'
}

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

#----------------------------------------
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred)) if np.all(y_pred > 0) else None
    mape_value = mape(y_true, y_pred)
    return mae, mse, rmse, r2, rmsle, mape_value

# Define regressors
regressors = {
    "Dummy Regressor": DummyRegressor(),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Elastic Net": ElasticNet(),
    "Bayesian Ridge": BayesianRidge(),
    "Orthogonal Matching Pursuit": OrthogonalMatchingPursuit(),
    "Huber Regressor": HuberRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "CatBoost Regressor": CatBoostRegressor(silent=True),
    #"Passive Aggressive Regressor": PassiveAggressiveRegressor(),
    "K Neighbors Regressor": KNeighborsRegressor(),
    "LGBM Regressor": LGBMRegressor(),
    "AdaBoost Regressor": AdaBoostRegressor(),
    "Extra Trees Regressor": ExtraTreesRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor()
}

def plot_learning_curve(model, X_train, y_train, title="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    plt.figure(figsize=(8, 3))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-Validation Score")
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Score (Negative MSE)")
    plt.legend(loc="best")
    st.pyplot(plt, use_container_width=True)

def plot_validation_curve(model, X_train, y_train, param_name, param_range, title="Validation Curve"):
    train_scores, val_scores = validation_curve(model, X_train, y_train, param_name=param_name, param_range=param_range, cv=5, scoring='neg_mean_squared_error')
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    plt.figure(figsize=(8, 3))
    plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(param_range, val_scores_mean, 'o-', color="g", label="Cross-Validation Score")
    plt.title(title)
    plt.xlabel(f"Values of {param_name}")
    plt.ylabel("Score (Negative MSE)")
    plt.legend(loc="best")
    st.pyplot(plt, use_container_width=True)

#----------------------------------------
clustering_algorithms = {
    "KMeans": KMeans(n_clusters=3),
    "AffinityPropagation": AffinityPropagation(),
    "MeanShift": MeanShift(),
    "SpectralClustering": SpectralClustering(n_clusters=3),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3),
    "DBSCAN": DBSCAN(),
    "OPTICS": OPTICS(),
    "Birch": Birch(n_clusters=3),
    "KModes": KModes(n_clusters=3, init='Cao', n_init=5, verbose=1)
}    

#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

ml_type = st.selectbox("", ["None", "Classification", "Clustering", "Regression",])                                                 
if ml_type == "None":
    st.warning("Please choose an algorithm to proceed with the analysis.")
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

else:
    
    #st.info("""The **Classification** tab xx.""")
    
    col1, col2= st.columns((0.2,0.8))
    with col1:           
        with st.container(border=True):
    
            file = st.file_uploader("**:blue[Choose a file]**",type=["csv", "xls", "xlsx"], accept_multiple_files=False, key="file_upload")
            if file is not None:
                st.success("Data loaded successfully!")
                df = load_file(file)
                
                st.divider()
                
                with st.popover("**:blue[:hammer_and_wrench: Cleaning Criteria]**",disabled=False, use_container_width=True,help="Tune the hyperparameters whenever required"):          
                        numerical_strategies = ['mean', 'median', 'most_frequent']
                        categorical_strategies = ['constant','most_frequent']
                        selected_numerical_strategy = st.selectbox("**Missing value treatment : Numerical**", numerical_strategies)
                        selected_categorical_strategy = st.selectbox("**Missing value treatment : Categorical**", categorical_strategies) 
                        st.divider() 
                        treatment_option = st.selectbox("**Select a outlier treatment option**", ["Cap Outliers","Drop Outliers", ])

                with st.popover("**:blue[:hammer_and_wrench: Transformation Criteria]**",disabled=False, use_container_width=True,help="Tune the hyperparameters whenever required"):
                        scaling_reqd = st.selectbox("**Requirement of scalling**", ["no", "yes"])
                        if scaling_reqd == 'yes':                       
                            scaling_method = st.selectbox("**Scaling method**", ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
                        if scaling_reqd == 'no':   
                            scaling_method = 'N/A'
                        st.divider()
                        f_sel_method = ['VIF', 'Selectkbest','VarianceThreshold']
                        f_sel_method = st.selectbox("**Feature selection method**", f_sel_method)
                        if f_sel_method == 'VIF':
                            vif_threshold = st.number_input("**VIF Threshold**", 1.5, 10.0, 5.0)                        
                        if f_sel_method == 'Selectkbest':
                            method = st.selectbox("**kBest Method**", ["f_classif", "f_regression", "chi2", "mutual_info_classif"])
                            num_features_to_select = st.slider("**Number of Independent Features**", min_value=1, max_value=len(df.columns), value=5)
                        if f_sel_method == 'VarianceThreshold':
                            threshold = st.number_input("Variance Threshold", min_value=0.0, step=0.01, value=0.0)      
                                       
                st.divider()
                target_variable = st.selectbox("**:blue[Choose Target Variable]**", options=["None"] + list(df.columns), key="target_variable")
                if target_variable == "None":
                    st.warning("Please choose a target variable to proceed with the analysis.")                
                
                else:
                    with col2:
                        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["**Information**","**Visualizations**","**Cleaning**","**Transformation**","**Performance**","**Graph**","**Results**",])
                        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        with tab1:
                            with st.container(border=True):
                      
                                st.table(df.head(2))
                                #--------------------------------------------
                                st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Basic Statistics</span></div>',unsafe_allow_html=True,)
                                #--------------------------------------------
                                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

                                col1.metric('**input values (rows)**', df.shape[0], help='number of rows')
                                col2.metric('**variables (columns)**', df.shape[1], help='number of columns')     
                                col3.metric('**numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
                                col4.metric('**categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')
                
                                col5.metric('**Missing values**', df.isnull().sum().sum(), help='Total missing values in the dataset')
                                #col6.metric('**Unique categorical values**', sum(df.select_dtypes(include=['object']).nunique()), help='Sum of unique values in categorical variables')
                                col6.metric('**Target Variable**', target_variable, help='Selected target variable')
                                
                                if ml_type == "Classification":
                                    unique_vals = df[target_variable].nunique()
                                    if unique_vals == 2:
                                        target_type = "Binary"
                                    else:
                                        target_type = "Multiclass"
                                    col7.metric('**Type of Target Variable**', target_type, help='Classification problem type (binary/multiclass)')       
                                else:
                                    col7.metric('**Type of Target Variable**', "None",)
                                #--------------------------------------------
                                st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Distribution</span></div>',unsafe_allow_html=True,)
                                #--------------------------------------------
                                plot_histograms_with_kde(df)

                                st.divider()
                                
                                plot_scatter(df, target_variable) 
                                
                        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        with tab2:
                            with st.container(border=True):
                                
                                st.write("xx")

                        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        with tab3:
                            with st.container(border=True):
                                
                                #--------------------------------------------
                                st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Missing Values</span></div>',unsafe_allow_html=True,)
                                #--------------------------------------------
                                col1, col2 = st.columns((0.2,0.8))
                                with col1:
                        
                                    missing_values = check_missing_values(df)
                                    if missing_values.empty:
                                        st.success("**No missing values found!**")
                                    else:
                                        st.warning("**Missing values found!**")
                                        st.write("**Number of missing values:**")
                                        st.table(missing_values)

                                        with col2:   
                                                          
                                            st.write("**Missing Values Treatment:**")                  
                                            cleaned_df = handle_numerical_missing_values(df, selected_numerical_strategy)
                                            cleaned_df = handle_categorical_missing_values(cleaned_df, selected_categorical_strategy)   
                                            st.table(cleaned_df.head(2))
                                            st.download_button("**Download Treated Data**", cleaned_df.to_csv(index=False), file_name="treated_data.csv")

                                #with col2:

                                #--------------------------------------------
                                st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Duplicate Values</span></div>',unsafe_allow_html=True,)
                                #--------------------------------------------
                                if st.checkbox("Show Duplicate Values"):
                                        if missing_values.empty:
                                            st.table(df[df.duplicated()].head(2))
                                        else:
                                            st.table(cleaned_df[cleaned_df.duplicated()].head(2))

                                #--------------------------------------------
                                st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Outliers</span></div>',unsafe_allow_html=True,)
                                #--------------------------------------------
                                if missing_values.empty:
                                    df = df.copy()
                                else:
                                    df = cleaned_df.copy()

                                col1, col2 = st.columns((0.2,0.8))
                                with col1:
                        
                                    outliers = check_outliers(df)
                                    if outliers.empty:
                                        st.success("No outliers found!")
                                    else:
                                        st.warning("**Outliers found!**")
                                        st.write("**Number of outliers:**")
                                        st.table(outliers)
                    
                                with col2:
                                    
                                    if treatment_option == "Drop Outliers":
                                        df = df[~outliers['Column'].isin(outliers[outliers['Number of Outliers'] > 0]['Column'])]
                                        st.success("Outliers dropped. Preview of the cleaned dataset:")
                                        st.write(df.head())
                                    
                                    elif treatment_option == "Cap Outliers":
                                        df = df.copy()
                                        for column in outliers['Column'].unique():
                                            Q1 = df[column].quantile(0.25)
                                            Q3 = df[column].quantile(0.75)
                                            IQR = Q3 - Q1
                                            threshold = 1.5
                                            df[column] = np.where(df[column] < Q1 - threshold * IQR, Q1 - threshold * IQR, df[column])
                                            df[column] = np.where(df[column] > Q3 + threshold * IQR, Q3 + threshold * IQR, df[column])
                                            st.success("Outliers capped. Preview of the capped dataset:")
                                            st.write(df.head())
                                            
                        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        with tab4:
                            with st.container(border=True):
                                
                                col1, col2 = st.columns((0.3,0.7))  
                                with col1:
                        
                                    #--------------------------------------------
                                    st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Feature Encoding</span></div>',unsafe_allow_html=True,)
                                    #--------------------------------------------

                                    categorical_columns = df.select_dtypes(include=['object']).columns
                                    if len(categorical_columns) == 0:
                                        st.info("There are no categorical variables in the dataset.Proceed with the original DataFrame")
                                        df = df.copy()
                                    else:
                                        for feature in df.columns: 
                                            if df[feature].dtype == 'object': 
                                                print('\n')
                                                print('feature:',feature)
                                                print(pd.Categorical(df[feature].unique()))
                                                print(pd.Categorical(df[feature].unique()).codes)
                                                df[feature] = pd.Categorical(df[feature]).codes
                                        st.info("Categorical variables are encoded")
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(label="📥 Download Encoded Data (for review)", data=csv, file_name='encoded_data.csv', mime='text/csv')

                                    #--------------------------------------------
                                    st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Feature Scalling</span></div>',unsafe_allow_html=True,)
                                    #--------------------------------------------
                                    if scaling_reqd == 'yes':     
                                        df = scale_features(df,scaling_method)
                                        st.info("Data is scaled for further treatment")
                                        csv = df.to_csv(index=False).encode('utf-8')
                                        st.download_button(label="📥 Download Scaled Data (for review)", data=csv, file_name='scaled_data.csv', mime='text/csv')
                                    else:
                                        st.info("Data is not scaled, orginal data is considered for further treatment")
                                    #st.dataframe(df.head())
                                    
                                with col2:   

                                    #--------------------------------------------
                                    st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Feature Selection</span></div>',unsafe_allow_html=True,)
                                    #--------------------------------------------                                       

                                    if f_sel_method == 'VIF':

                                        st.markdown("**Method 1 : VIF**")

                                        st.markdown(f"Iterative VIF Thresholding (Threshold: {vif_threshold})")
                                        X = df.drop(columns = target_variable)
                                        vif_data = drop_high_vif_variables(df, vif_threshold)
                                        vif_data = vif_data.drop(columns = target_variable)
                                        selected_features = vif_data.columns
                                        st.markdown("**Selected Features (considering VIF values in ascending orders)**")
                                        st.write("No of features before feature-selection :",df.shape[1])
                                        st.write("No of features after feature-selection :",len(selected_features))
                                        st.dataframe(selected_features, hide_index=True)
                                        #st.table(vif_data)

                                    if f_sel_method == 'Selectkbest':
                  
                                        st.markdown("**Method 2 : Selectkbest**")          
                            
                                        if "f_classif" in method:
                                            feature_selector = SelectKBest(score_func=f_classif, k=num_features_to_select)
                                        elif "f_regression" in method:
                                            feature_selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
                                        elif "chi2" in method:
                                            df[df < 0] = 0
                                            feature_selector = SelectKBest(score_func=chi2, k=num_features_to_select)
                                        elif "mutual_info_classif" in method:
                                            df[df < 0] = 0
                                            feature_selector = SelectKBest(score_func=mutual_info_classif, k=num_features_to_select)

                                        X = df.drop(columns = target_variable)  
                                        y = df[target_variable]  
                                        X_selected = feature_selector.fit_transform(X, y)

                                        selected_feature_indices = feature_selector.get_support(indices=True)
                                        selected_features_kbest = X.columns[selected_feature_indices]
                                        st.markdown("**Selected Features (considering values in 'recursive feature elimination' method)**")
                                        st.write("No of features before feature-selection :",df.shape[1])
                                        st.write("No of features after feature-selection :",len(selected_features_kbest))
                                        st.dataframe(selected_features_kbest, hide_index=True)
                                        selected_features = selected_features_kbest.copy()

                                    if f_sel_method == 'VarianceThreshold':

                                        st.markdown("**Method 3 : VarianceThreshold**")  

                                        X = df.drop(columns = target_variable)  
                                        y = df[target_variable]
                                        selector = VarianceThreshold(threshold=threshold)
                                        X_selected = selector.fit_transform(X)

                                        selected_feature_indices = selector.get_support(indices=True)
                                        selected_features_vth = X.columns[selected_feature_indices]          
                                        st.markdown("**Selected Features (considering values in 'variance threshold' method)**") 
                                        st.write("No of features before feature-selection :",df.shape[1])
                                        st.write("No of features after feature-selection :",len(selected_features_vth))                   
                                        st.dataframe(selected_features_vth, hide_index=True)
                                        selected_features = selected_features_vth.copy()
