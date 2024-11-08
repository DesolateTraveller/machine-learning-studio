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
import scikitplot as skplt
#----------------------------------------
import shutil
import sweetviz as sv
import pygwalker as pyg
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
st.set_page_config(page_title="ML Studio | v0.2",
                   layout="wide",
                   page_icon="💻",              
                   initial_sidebar_state="auto")
#----------------------------------------
st.title(f""":rainbow[Machine Learning (ML) Studio]""")
#st.markdown(
    #'''
    #Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a> ( :envelope: [Email](mailto:avijit.mba18@gmail.com) | :bust_in_silhouette: [LinkedIn](https://www.linkedin.com/in/avijit2403/) | :computer: [GitHub](https://github.com/DesolateTraveller) ) |
    #for best view of the app, please **zoom-out** the browser to **75%**.
    #''',
    #unsafe_allow_html=True)
#st.info('**A lightweight Machine Learning (ML) streamlit app that help to analyse different kind machine learning problems**', icon="ℹ️")
#----------------------------------------
# Set the background image
#st.divider()
st.info('**A lightweight Machine Learning (ML) streamlit app that help to analyse different kind machine learning problems**', icon="ℹ️")
#----------------------------------------
st.sidebar.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
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
        <p>© 2024 | Created by : <span class="highlight">Avijit Chakraborty</span> | Prepared by: <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a></p> <span class="highlight">Thank you for visiting the app | This app is created for internal use, unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span>
    </div>
    """,
    unsafe_allow_html=True)
#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

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
#st.sidebar.header("Input", divider='blue')
#st.sidebar.info('Please choose from the following options to start the application.', icon="ℹ️")
ml_type = st.sidebar.selectbox("**:blue[Pick your Problem Type]**", ["None", "Classification", "Clustering", "Regression",])
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------                                                   
if ml_type == "None":
        st.warning("Please choose an algorithm in the sidebar to proceed with the analysis.")
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
else:        
    file = st.sidebar.file_uploader("**:blue[Choose a file]**",
                                    type=["csv", "xls", "xlsx"], 
                                    accept_multiple_files=False, 
                                    key="file_upload")
    if file is not None:
        df = load_file(file)
        st.sidebar.divider()

        stats_expander = st.expander("**Preview of Data**", expanded=True)
        with stats_expander:  
            st.table(df.head(2))
        st.divider()

        target_variable = st.sidebar.selectbox("**:blue[Choose Target Variable]**", options=["None"] + list(df.columns), key="target_variable")
        st.sidebar.divider()
        if target_variable == "None":
            st.warning("Please choose a target variable to proceed with the analysis.")

#---------------------------------------------------------------------------------------------------------------------------------
        else:  
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["**Information**","**Visualizations**","**Cleaning**","**Transformation**","**Performance**","**Graph**","**Results**",])
            
#---------------------------------------------------------------------------------------------------------------------------------
            with tab1:

                #st.subheader("**Data Analysis**",divider='blue')
                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

                col1.metric('**input values (rows)**', df.shape[0], help='number of rows')
                col2.metric('**variables (columns)**', df.shape[1], help='number of columns')     
                col3.metric('**numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
                col4.metric('**categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')
                
                col5.metric('**Missing values**', df.isnull().sum().sum(), help='Total missing values in the dataset')
                #col6.metric('**Unique categorical values**', sum(df.select_dtypes(include=['object']).nunique()), help='Sum of unique values in categorical variables')
                col6.metric('**Target Variable**', target_variable, help='Selected target variable')

                # Determine if it's a binary or multiclass classification
                if ml_type == "Classification":
                    unique_vals = df[target_variable].nunique()
                    if unique_vals == 2:
                        target_type = "Binary"
                    else:
                        target_type = "Multiclass"
                    col7.metric('**Type of Target Variable**', target_type, help='Classification problem type (binary/multiclass)')
                else:
                    col7.metric('**Type of Target Variable**', "None",)
                #st.divider()           

                stats_expander = st.expander("**Exploratory Data Analysis (EDA)**", expanded=False)
                with stats_expander:        
                    #pr = df.profile_report()
                    #st_profile_report(pr)
                    st.table(df.head()) 

#---------------------------------------------------------------------------------------------------------------------------------
            with tab2:

                plot_option = st.selectbox("**Choose Plot**", ["Line Chart", "Histogram", "Scatter Plot", "Bar Chart", "Box Plot"])
                columns = list(df.columns)
                col1, col2 = st.columns((0.1,0.9))
                    
                if plot_option == "Line Chart":

                    with col1:
                            x_column = st.selectbox("**:blue[Select X column]**", options=columns, key="date_1", )
                            y_column = st.selectbox("**:blue[Select Y column]**", options=columns, key="values_1")
                        
                    with col2:
                            line_chart = alt.Chart(df).mark_line().encode(
                            x=alt.X(x_column, type='temporal' if pd.api.types.is_datetime64_any_dtype(df[x_column]) else 'ordinal'),
                            y=alt.Y(y_column, type='quantitative'),
                            tooltip=[x_column, y_column]).interactive()
                            st.altair_chart(line_chart, use_container_width=True)

                elif plot_option == "Histogram":
                        
                    with col1:
                            x_column = st.selectbox("**:blue[Select column for histogram]**", options=columns, key="hist_1",)
                        
                    with col2:
                            histogram = alt.Chart(df).mark_bar().encode(
                            x=alt.X(x_column, bin=True),
                            y=alt.Y('count()', type='quantitative'),
                            tooltip=[x_column, 'count()']).interactive()
                            st.altair_chart(histogram, use_container_width=True)

                elif plot_option == "Scatter Plot":
                        
                    with col1:
                            x_column = st.selectbox("**:blue[Select X column]**", options=columns, key="scatter_x", )
                            y_column = st.selectbox("**:blue[Select Y column]**", options=columns, key="scatter_y", )
                        
                    with col2:
                            scatter_plot = alt.Chart(df).mark_point().encode(
                            x=alt.X(x_column, type='quantitative' if pd.api.types.is_numeric_dtype(df[x_column]) else 'ordinal'),
                            y=alt.Y(y_column, type='quantitative'),
                            tooltip=[x_column, y_column]).interactive()
                            st.altair_chart(scatter_plot, use_container_width=True)

                elif plot_option == "Bar Chart":
                    
                    with col1:
                            x_column = st.selectbox("**:blue[Select X column]**", options=columns, key="bar_x", )
                            y_column = st.selectbox("**:blue[Select Y column]**", options=columns, key="bar_y", )
                        
                    with col2:
                            bar_chart = alt.Chart(df).mark_bar().encode(
                            x=alt.X(x_column, type='ordinal' if not pd.api.types.is_numeric_dtype(df[x_column]) else 'quantitative'),
                            y=alt.Y(y_column, type='quantitative'),
                            tooltip=[x_column, y_column]).interactive()
                            st.altair_chart(bar_chart, use_container_width=True)

                elif plot_option == "Box Plot":
                    
                    with col1:
                            x_column = st.selectbox("**:blue[Select X column]**", options=columns, key="box_x",)
                            y_column = st.selectbox("**:blue[Select Y column]**", options=columns, key="box_y", )
                        
                    with col2:
                            box_plot = alt.Chart(df).mark_boxplot().encode(
                            x=alt.X(x_column, type='ordinal' if not pd.api.types.is_numeric_dtype(df[x_column]) else 'quantitative'),
                            y=alt.Y(y_column, type='quantitative'),
                            tooltip=[x_column, y_column]).interactive()
                            st.altair_chart(box_plot, use_container_width=True)

#---------------------------------------------------------------------------------------------------------------------------------
            with tab3:

                    #stats_expander = st.sidebar.expander("**:blue[Cleaning Criteria]**", expanded=False)
                    #with stats_expander:       
                    with st.sidebar.popover("**:blue[:hammer_and_wrench: Cleaning Criteria]**", help="Tune the hyperparameters whenever required"):          
                        numerical_strategies = ['mean', 'median', 'most_frequent']
                        categorical_strategies = ['constant','most_frequent']
                        selected_numerical_strategy = st.selectbox("**Missing value treatment : Numerical**", numerical_strategies)
                        selected_categorical_strategy = st.selectbox("**Missing value treatment : Categorical**", categorical_strategies) 
                        st.divider() 
                        treatment_option = st.selectbox("**Select a outlier treatment option**", ["Cap Outliers","Drop Outliers", ])

                    st.subheader("Missing Values",divider='blue')
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

                    st.subheader("Duplicate Values",divider='blue') 
                    if st.checkbox("Show Duplicate Values"):
                        if missing_values.empty:
                            st.table(df[df.duplicated()].head(2))
                        else:
                            st.table(cleaned_df[cleaned_df.duplicated()].head(2))

                    st.subheader("Outliers",divider='blue')
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

#---------------------------------------------------------------------------------------------------------------------------------
            with tab4:

                    #stats_expander = st.sidebar.expander("**:blue[Transformation Criteria]**", expanded=False)
                    #with stats_expander: 
                    with st.sidebar.popover("**:blue[:hammer_and_wrench: Transformation Criteria]**", help="Tune the hyperparameters whenever required"):
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

                    col1, col2 = st.columns((0.3,0.7))  

                    with col1:
                        
                        st.subheader("Feature Encoding",divider='blue') 

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

                        st.divider()

                        st.subheader("Feature Scaling",divider='blue')
                        if scaling_reqd == 'yes':     
                            df = scale_features(df,scaling_method)
                            st.info("Data is scaled for further treatment")
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(label="📥 Download Scaled Data (for review)", data=csv, file_name='scaled_data.csv', mime='text/csv')
                        else:
                            st.info("Data is not scaled, orginal data is considered for further treatment")
                        #st.dataframe(df.head())

                    #----------------------------------------

                    with col2:   

                        st.subheader("Feature Selection",divider='blue')                                         

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
                            st.table(selected_features)
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
                            st.table(selected_features_kbest)
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
                            st.table(selected_features_vth)
                            selected_features = selected_features_vth.copy()

#---------------------------------------------------------------------------------------------------------------------------------
            with tab5:

                st.info("Please note that there may be some processing delay during the AutoML execution.")
                #st.sidebar.divider()

                #stats_expander = st.sidebar.expander("**:blue[Dataset Splitting Criteria]**", expanded=False)
                #with stats_expander:
                with st.sidebar.popover("**:blue[Dataset Splitting Criteria]**", help="Tune the hyperparameters whenever required"):   
                        train_size = st.slider("**Train Size (as %)**", 10, 90, 70, 5)
                        test_size = st.slider("**Test Size (as %)**", 10, 50, 30, 5)    
                        random_state = st.number_input("**Random State**", 0, 100, 42)
                        n_jobs = st.number_input("**Parallel Processing (n_jobs)**", -10, 10, 1)    

                #stats_expander = st.sidebar.expander("**:blue[:hammer_and_wrench: Hyperparameters]**", expanded=False)
                #with stats_expander:
                with st.sidebar.popover("**:blue[:hammer_and_wrench: Hyperparameters]**", help="Tune the hyperparameters whenever required"):
                        n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, step=10, value=100)
                        max_depth = st.slider("Max Depth", min_value=1, max_value=20, step=1, value=10)    
                        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, step=1, value=2)
                        learning_rate = st.number_input("Learning rate", .01, .1, step =.01, key ='learning_rate')
                        C = st.slider("C (Regularization)", min_value=0.01, max_value=10.0, step=0.01, value=1.0)   
                        kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                        gamma = st.selectbox("Gamma", ["scale", "auto"])
                        solver= st.radio("**Solver**", ('liblinear', 'lbfgs'))
                        penalty = st.selectbox("Penalty", ["l1", "l2", "elasticnet"])

                X = df[selected_features]
                y = df[target_variable]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            #----------------------------------------                 
                if ml_type == 'Classification': 

                    #clf_typ = st.sidebar.selectbox("**:blue[Choose the type of target]**", ["Binary", "MultiClass"]) 
                    models = {
                        "Logistic Regression": LogisticRegression(penalty=penalty, C=C, solver=solver),
                        "Ridge Classifier": RidgeClassifier(),
                        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                        "Random Forest Classifier": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split),
                        #"Naive Bayes": GaussianNB(),
                        #"CatBoost Classifier": CatBoostClassifier(verbose=0),
                        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate),
                        "Ada Boost Classifier": AdaBoostClassifier(),
                        "Extra Trees Classifier": ExtraTreesClassifier(),
                        #"Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                        "Light Gradient Boosting Machine": LGBMClassifier(),
                        "K Neighbors Classifier": KNeighborsClassifier(),
                        "Decision Tree Classifier": DecisionTreeClassifier(),
                        #"Extreme Gradient Boosting": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
                        #"SVM - Linear Kernel": SVC(kernel="linear", probability=True)
                        }
                    #----------------------------------------
                    if target_type == "Binary":
                        #if st.sidebar.button("Submit"):

                            col1, col2 = st.columns((0.4,0.6))  
                            with col1:
                                    
                                with st.container():

                                    st.subheader("Comparison",divider='blue')
                                    with st.spinner("Setting up and comparing models..."):

                                        results = []
                                        for name, model in models.items():
                                            metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
                                            metrics["Model"] = name
                                            results.append(metrics)
                                        results_df = pd.DataFrame(results)
                                        best_metrics = results_df.loc[:, results_df.columns != "Model"].idxmax()
                                        #st.dataframe(results_df,hide_index=True, use_container_width=True)
                                        st.table(results_df)

                                        best_model_clf = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
                                        best_model = models[best_model_clf]
                                        best_model.fit(X_train, y_train)
                                        y_pred_best = best_model.predict(X_test)
                                        y_proba_best = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None                                        
                                        #st.sidebar.info(f"Best model : **{best_model_clf}**")

                            with col2:
                                    
                                with st.container():

                                    st.subheader("Graph",divider='blue')
                                    metrics_df = results_df.melt(id_vars="Model", value_vars=["Accuracy", "AUC", "Recall", "Precision", "F1 Score", "Kappa", "MCC"], var_name="Metric", value_name="Value")
                                    plt.figure(figsize=(10,6))
                                    sns.barplot(x="Metric", y="Value", hue="Model", data=metrics_df, palette="rocket")
                                    plt.title("Comparison of Classification Metrics Across Models", fontsize=16)
                                    plt.xlabel("Metric", fontsize=12)
                                    plt.ylabel("Value", fontsize=12)
                                    plt.xticks(rotation=45)
                                    plt.legend(title="Model", bbox_to_anchor=(1.05,1), loc='upper left')
                                    st.pyplot(plt, use_container_width=True)
    
                                    #st.subheader("Hyperparameters",divider='blue')
                                    #if isinstance(best_model, LogisticRegression):
                                        #st.write(f"**C**: {C}")
                                    #elif isinstance(best_model, RandomForestClassifier):
                                        #st.write(f"**n_estimators**: {n_estimators}")
                                        #st.write(f"**max_depth**: {max_depth}")
                                        #st.write(f"**min_samples_split**: {min_samples_split}")
                                    #elif isinstance(best_model, GradientBoostingClassifier):
                                        #st.write(f"**n_estimators**: {n_estimators}")
                                        #st.write(f"**learning_rate**: {learning_rate}")
                                    #elif isinstance(best_model, LGBMClassifier):
                                        #st.write(f"**n_estimators**: {n_estimators}")
                                        #st.write(f"**learning_rate**: {learning_rate}")

                    #----------------------------------------
                    elif target_type == "MultiClass":
                            
                            col1, col2 = st.columns((0.4,0.6))  
                            with col1:
                                
                                with st.container():
                                    
                                    st.subheader("Comparison",divider='blue')
                                    with st.spinner("Setting up and comparing models..."):
                
                                        results = []
                                        for name, model in models.items():
                                            metrics = evaluate_model(model, X_train, X_test, y_train, y_test, multi_class=True)
                                            metrics["Model"] = name
                                            results.append(metrics)
                                        results_df = pd.DataFrame(results)
                                        best_metrics = results_df.loc[:, results_df.columns != "Model"].idxmax()
                                        #st.dataframe(results_df,hide_index=True, use_container_width=True)
                                        st.table(results_df)

                                        best_model_clf = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
                                        best_model = models[best_model_clf]
                                        best_model.fit(X_train, y_train)
                                        y_pred_best = best_model.predict(X_test)
                                        y_proba_best = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
                                        #st.sidebar.info(f"Best model : **{best_model_clf}**")

                            with col2:

                                with st.container():

                                    st.subheader("Graph",divider='blue')
                                    metrics_df = results_df.melt(id_vars="Model", value_vars=["Accuracy", "AUC", "Recall", "Precision", "F1 Score", "Kappa", "MCC"], var_name="Metric", value_name="Value")
                                    plt.figure(figsize=(10,6))
                                    sns.barplot(x="Metric", y="Value", hue="Model", data=metrics_df, palette="rocket")
                                    plt.title("Comparison of Classification Metrics Across Models", fontsize=16)
                                    plt.xlabel("Metric", fontsize=12)
                                    plt.ylabel("Value", fontsize=12)
                                    plt.xticks(rotation=45)
                                    plt.legend(title="Model", bbox_to_anchor=(1.05,1), loc='upper left')
                                    st.pyplot(plt, use_container_width=True)

                                    #st.subheader("Hyperparameters",divider='blue')
                                    #if isinstance(best_model, LogisticRegression):
                                        #st.write(f"**C**: {C}")
                                    #elif isinstance(best_model, RandomForestClassifier):
                                        #st.write(f"**n_estimators**: {n_estimators}")
                                        #st.write(f"**max_depth**: {max_depth}")
                                        #st.write(f"**min_samples_split**: {min_samples_split}")
                                    #elif isinstance(best_model, GradientBoostingClassifier):
                                        #st.write(f"**n_estimators**: {n_estimators}")
                                        #st.write(f"**learning_rate**: {learning_rate}")
                                    #elif isinstance(best_model, LGBMClassifier):
                                        #st.write(f"**n_estimators**: {n_estimators}")
                                        #st.write(f"**learning_rate**: {learning_rate}")

                    #----------------------------------------                    
                    st.divider()
                    st.subheader("Importance",divider='blue')

                    if best_model_clf == "Logistic Regression":
                        #importance = best_model.coef_.flatten()
                        importance_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': best_model.coef_.flatten()})
                        importance_df['Percentage'] = (abs(importance_df['Coefficient']) / abs(importance_df['Coefficient']).sum()) * 100
                        importance_df = importance_df.sort_values(by='Coefficient', ascending=False)
                    else:
                        #importance = best_model.feature_importances_
                        importance_df = pd.DataFrame({"Feature": selected_features, "Importance": best_model.feature_importances_})
                        importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
                        importance_df = importance_df.sort_values(by='Importance', ascending=False)

                    col1, col2 = st.columns((0.25,0.75))
                    with col1:
                        with st.container():

                            #importance_df = pd.DataFrame({"Feature": selected_features,"Importance": importance})
                            st.dataframe(importance_df, hide_index=True, use_container_width=True)

                    with col2:
                        with st.container():
                                        
                            plot_data_imp = [go.Bar(x = importance_df['Feature'],y = importance_df['Importance'])]
                            plot_layout_imp = go.Layout(xaxis = {"title": "Feature"},yaxis = {"title": "Importance"},title = 'Feature Importance',)
                            fig = go.Figure(data = plot_data_imp, layout = plot_layout_imp)
                            st.plotly_chart(fig,use_container_width = True)

                    st.sidebar.divider()
                    st.sidebar.info(f"Best model : **{best_model_clf}**")
                    if best_model_clf == "Logistic Regression":
                        stats_expander = st.sidebar.expander("**:blue[Important features]**", expanded=False)
                        with stats_expander:
                            #st.info(f"**Top Features based on Importance:**\n1. {importance_df.iloc[0]['Feature']}\n2. {importance_df.iloc[1]['Feature']}\n3. {importance_df.iloc[2]['Feature']}\n4. {importance_df.iloc[3]['Feature']}\n5. {importance_df.iloc[4]['Feature']}")
                            st.info(f"**Top Features based on Importance:**\n"
                                    f"1. {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Percentage']:.2f}%)\n"
                                    f"2. {importance_df.iloc[1]['Feature']} ({importance_df.iloc[1]['Percentage']:.2f}%)\n"
                                    f"3. {importance_df.iloc[2]['Feature']} ({importance_df.iloc[2]['Percentage']:.2f}%)\n"
                                    f"4. {importance_df.iloc[3]['Feature']} ({importance_df.iloc[3]['Percentage']:.2f}%)\n"
                                    f"5. {importance_df.iloc[4]['Feature']} ({importance_df.iloc[4]['Percentage']:.2f}%)"
                                        )           
                    else:
                        stats_expander = st.sidebar.expander("**:blue[Important features]**", expanded=False)
                        with stats_expander:
                            #st.info(f"**Top Features based on Importance:**\n1. {importance_df.iloc[0]['Feature']}\n2. {importance_df.iloc[1]['Feature']}\n3. {importance_df.iloc[2]['Feature']}\n4. {importance_df.iloc[3]['Feature']}\n5. {importance_df.iloc[4]['Feature']}")
                            st.info(f"**Top Features based on Importance:**\n"
                                    f"1. {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Percentage']:.2f}%)\n"
                                    f"2. {importance_df.iloc[1]['Feature']} ({importance_df.iloc[1]['Percentage']:.2f}%)\n"
                                    f"3. {importance_df.iloc[2]['Feature']} ({importance_df.iloc[2]['Percentage']:.2f}%)\n"
                                    f"4. {importance_df.iloc[3]['Feature']} ({importance_df.iloc[3]['Percentage']:.2f}%)\n"
                                    f"5. {importance_df.iloc[4]['Feature']} ({importance_df.iloc[4]['Percentage']:.2f}%)"
                                        )
            #----------------------------------------                
                if ml_type == 'Regression': 

                    col1, col2 = st.columns((0.4,0.6))  
                    with col1:
                                
                        with st.container():
                                    
                            st.subheader("Comparison",divider='blue')
                            with st.spinner("Setting up and comparing models..."):

                                results = []
                                for name, model in regressors.items():
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    mae, mse, rmse, r2, rmsle, mape_value = calculate_metrics(y_test, y_pred)
                                    results.append({"Model": name,
                                            "MAE": round(mae, 2),
                                            "MSE": round(mse, 2),
                                            "RMSE": round(rmse, 2),
                                            "R2": round(r2, 2),
                                            "RMSLE": round(rmsle, 2) if rmsle else "N/A",
                                            "MAPE": round(mape_value, 2)})
                                results_df = pd.DataFrame(results)
                                #st.dataframe(results_df,hide_index=True, use_container_width=True)
                                st.table(results_df)
                                
                                best_model_reg = results_df.loc[results_df['R2'].idxmax(), 'Model']
                                #st.sidebar.info(f"Best model : **{best_model_reg}**")
                                best_model = regressors[best_model_reg]
                                y_pred_best = best_model.predict(X_test)
                                residuals = y_test - y_pred_best    

                    with col2:
                                
                        with st.container():

                            st.subheader("Graph",divider='blue')
                            metrics_df = results_df.melt(id_vars="Model", value_vars=["MAE", "MSE", "RMSE", "R2", "RMSLE", "MAPE"], 
                                                        var_name="Metric", value_name="Value")
                            plt.figure(figsize=(10,6))
                            sns.barplot(x="Metric", y="Value", hue="Model", data=metrics_df, palette="rocket")
                            plt.title("Comparison of Regression Metrics Across Models", fontsize=16)
                            plt.xlabel("Metric", fontsize=12)
                            plt.ylabel("Value", fontsize=12)
                            plt.xticks(rotation=45)
                            plt.legend(title="Model", bbox_to_anchor=(1.05,1), loc='upper left')
                            st.pyplot(plt, use_container_width=True)

                    #----------------------------------------  
                    st.subheader("Importance",divider='blue')

                    if best_model_reg == "Linear Regression":
                        importance_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': best_model.coef_[0]})
                        importance_df['Percentage'] = (abs(importance_df['Coefficient']) / abs(importance_df['Coefficient']).sum()) * 100
                        importance_df = importance_df.sort_values(by='Coefficient', ascending=False)

                    else:
                        #importance = best_model.feature_importances_
                        importance_df = pd.DataFrame({"Feature": selected_features, "Importance": best_model.feature_importances_})
                        importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
                        importance_df = importance_df.sort_values(by='Importance', ascending=False)

                    col1, col2 = st.columns((0.25,0.75))
                    with col1:
                        with st.container():

                            #importance_df = pd.DataFrame({"Feature": selected_features,"Importance": importance})
                            st.dataframe(importance_df, hide_index=True, use_container_width=True)

                    with col2:
                        with st.container():
                                        
                            plot_data_imp = [go.Bar(x = importance_df['Feature'],y = importance_df['Importance'])]
                            plot_layout_imp = go.Layout(xaxis = {"title": "Feature"},yaxis = {"title": "Importance"},title = 'Feature Importance',)
                            fig = go.Figure(data = plot_data_imp, layout = plot_layout_imp)
                            st.plotly_chart(fig,use_container_width = True)

                    st.sidebar.divider()
                    st.sidebar.info(f"Best model : **{best_model_reg}**")
                    if best_model_reg == "Linear Regression":
                        stats_expander = st.sidebar.expander("**:blue[Important features]**", expanded=False)
                        with stats_expander:
                            #st.info(f"**Top Features based on Importance:**\n1. {importance_df.iloc[0]['Feature']}\n2. {importance_df.iloc[1]['Feature']}\n3. {importance_df.iloc[2]['Feature']}\n4. {importance_df.iloc[3]['Feature']}\n5. {importance_df.iloc[4]['Feature']}")
                            st.info(f"**Top Features based on Importance:**\n"
                                    f"1. {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Percentage']:.2f}%)\n"
                                    f"2. {importance_df.iloc[1]['Feature']} ({importance_df.iloc[1]['Percentage']:.2f}%)\n"
                                    f"3. {importance_df.iloc[2]['Feature']} ({importance_df.iloc[2]['Percentage']:.2f}%)\n"
                                    f"4. {importance_df.iloc[3]['Feature']} ({importance_df.iloc[3]['Percentage']:.2f}%)\n"
                                    f"5. {importance_df.iloc[4]['Feature']} ({importance_df.iloc[4]['Percentage']:.2f}%)"
                                        )           
                    else:
                        stats_expander = st.sidebar.expander("**:blue[Important features]**", expanded=False)
                        with stats_expander:
                            #st.info(f"**Top Features based on Importance:**\n1. {importance_df.iloc[0]['Feature']}\n2. {importance_df.iloc[1]['Feature']}\n3. {importance_df.iloc[2]['Feature']}\n4. {importance_df.iloc[3]['Feature']}\n5. {importance_df.iloc[4]['Feature']}")
                            st.info(f"**Top Features based on Importance:**\n"
                                    f"1. {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Percentage']:.2f}%)\n"
                                    f"2. {importance_df.iloc[1]['Feature']} ({importance_df.iloc[1]['Percentage']:.2f}%)\n"
                                    f"3. {importance_df.iloc[2]['Feature']} ({importance_df.iloc[2]['Percentage']:.2f}%)\n"
                                    f"4. {importance_df.iloc[3]['Feature']} ({importance_df.iloc[3]['Percentage']:.2f}%)\n"
                                    f"5. {importance_df.iloc[4]['Feature']} ({importance_df.iloc[4]['Percentage']:.2f}%)"
                                        )
            #----------------------------------------              
                if ml_type == 'Clustering': 

                    col1, col2 = st.columns((0.4,0.6))  
                    with col1:
                                
                        with st.container():
                                    
                            st.subheader("Comparison",divider='blue')
                            with st.spinner("Setting up and comparing models..."):

                                results = []
                                for name, algorithm in clustering_algorithms.items():
                                    try:
                                        if name == "KModes":
                                            labels = algorithm.fit_predict(X)
                                        else:
                                            labels = algorithm.fit_predict(X)
        
                                        silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else None
                                        calinski = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else None
                                        davies = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else None
                                        homogeneity = homogeneity_score(df[target_variable], labels)
                                        rand_index = adjusted_rand_score(df[target_variable], labels)
                                        completeness = completeness_score(df[target_variable], labels)
        
                                        results.append({"Algorithm": name,
                                                        "Silhouette": silhouette,
                                                        "Calinski-Harabasz": calinski,
                                                        "Davies-Bouldin": davies,
                                                        "Homogeneity": homogeneity,
                                                        "Rand Index": rand_index,
                                                        "Completeness": completeness})
                                    except Exception as e:
                                        print(f"Algorithm {name} failed: {e}")

                                results_df = pd.DataFrame(results)
                                st.dataframe(results_df,hide_index=True, use_container_width=True)                      

                                st.divider()
                                
                                best_model_clust = results_df.loc[results_df['Silhouette'].idxmax(), 'Algorithm']
                                st.sidebar.info(f"Best model : **{best_model_clust}**")
                                best_model = clustering_algorithms[best_model_clust]

#---------------------------------------------------------------------------------------------------------------------------------
            with tab6:
               
                if ml_type == 'Classification':        
                    
                    if target_type == 'Binary':

                        col1, col2 = st.columns(2)  
                        with col1:
                            with st.container():     
                                                           
                                    report = classification_report(y_test, y_pred_best, output_dict=True)
                                    report_df = pd.DataFrame(report).transpose()
                                    #st.dataframe(report_df,use_container_width=True)
                                    st.table(report_df)
                        
                        with col2:
                            with st.container():  

                                    cm = confusion_matrix(y_test, y_pred_best)
                                    plt.figure(figsize=(8,3))
                                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                                    plt.title(f"Confusion Matrix", fontsize=8)
                                    plt.xlabel("Predicted")
                                    plt.ylabel("Actual")
                                    st.pyplot(plt,use_container_width=True)

                        st.divider()           
                        
                        col1, col2 = st.columns(2)  
                        with col1: 
                            with st.container():
                                    
                                    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
                                    plt.figure(figsize=(8,3))
                                    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {auc(fpr, tpr):.2f}")
                                    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
                                    plt.xlabel("False Positive Rate")
                                    plt.ylabel("True Positive Rate")
                                    plt.title(f"AUC Curve", fontsize=8)
                                    plt.legend(loc="lower right")
                                    st.pyplot(plt,use_container_width=True)

                                    precisions, recalls, _ = precision_recall_curve(y_test, y_proba_best)
                                    plt.figure(figsize=(8,3))
                                    plt.plot(recalls, precisions, color="purple", lw=2)
                                    plt.xlabel("Recall")
                                    plt.ylabel("Precision")
                                    plt.title(f"Precision-Recall Curve", fontsize=8)
                                    st.pyplot(plt,use_container_width=True)

                                    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_best)
                                    plt.figure(figsize=(8,3))
                                    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
                                    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
                                    plt.xlabel("Threshold")
                                    plt.title(f"Discrimination Threshold", fontsize=8)
                                    plt.legend(loc="best")
                                    st.pyplot(plt,use_container_width=True)

                        with col2:
                            with st.container():

                                    plt.figure(figsize=(8,3))
                                    skplt.metrics.plot_lift_curve(y_test, best_model.predict_proba(X_test))
                                    plt.title(f"Lift Curve", fontsize=8)
                                    st.pyplot(plt,use_container_width=True)
                                    
                                    plt.figure(figsize=(8,3))
                                    skplt.metrics.plot_cumulative_gain(y_test, best_model.predict_proba(X_test))
                                    plt.title(f"Gain Curve", fontsize=8)
                                    st.pyplot(plt,use_container_width=True) 

                    if target_type == 'MultiClass':

                        col1, col2 = st.columns(2)  
                        with col1:
                            with st.container():  

                                    report = classification_report(y_test, y_pred_best, output_dict=True)
                                    report_df = pd.DataFrame(report).transpose()
                                    #st.dataframe(report_df,use_container_width=True)  
                                    st.table(report_df)

                        with col2:
                            with st.container():  

                                    cm = confusion_matrix(y_test, y_pred_best)
                                    plt.figure(figsize=(8,3))
                                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                                    plt.title(f"Confusion Matrix for {best_model_clf}", fontsize=8)
                                    plt.xlabel("Predicted")
                                    plt.ylabel("Actual")
                                    st.pyplot(plt,use_container_width=True)

            #----------------------------------------                
                if ml_type == 'Regression': 

                        col1, col2 = st.columns(2)  
                        with col1:
                            with st.container():                     
                                                      
                                    plt.figure(figsize=(8, 3))
                                    sns.residplot(x=y_pred_best, y=residuals, lowess=True)
                                    plt.title(f"Residual Plot")
                                    plt.xlabel('Predicted')
                                    plt.ylabel('Residuals')
                                    st.pyplot(plt,use_container_width=True)
    
                                    plt.figure(figsize=(8, 3))
                                    sns.scatterplot(x=y_test, y=y_pred_best)
                                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
                                    plt.title(f"Prediction Error Plot")
                                    plt.xlabel('Actual')
                                    plt.ylabel('Predicted')
                                    st.pyplot(plt,use_container_width=True) 

                        with col2:
                            with st.container(): 

                                    plot_learning_curve(best_model, X_train, y_train)  

                                    param_name = 'alpha'  
                                    param_range = np.logspace(-3, 3, 10)
                                    plot_validation_curve(best_model, X_train, y_train, param_name, param_range)

            #----------------------------------------                
                if ml_type == 'Clustering': 
                                
                    best_labels = best_model.fit_predict(X)
                    df['Cluster_Labels'] = best_labels
                    plt.figure(figsize=(8, 3))
                    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=best_labels, palette="viridis")
                    plt.title(f"Cluster plot for {best_model_clust}")
                    plt.show()
                    st.pyplot(plt,use_container_width = True)      

                    st.divider()

                    if "KMeans" in clustering_algorithms:
                                    inertia_values = []
                                    K_range = range(1, 11)
                                    for k in K_range:
                                        kmeans = KMeans(n_clusters=k)
                                        kmeans.fit(X)
                                        inertia_values.append(kmeans.inertia_)

                                    col1, col2 = st.columns((0.2,0.8))  
                                    with col1:  
                                        with st.container(): 

                                            elbow_df = pd.DataFrame({'K': K_range,'Inertia': inertia_values})  
                                            st.dataframe(elbow_df,hide_index=True, use_container_width=True)

                                            with col2:  
                                                with st.container(): 
                                                    
                                                    plt.figure(figsize=(8,3))
                                                    plt.plot(K_range, inertia_values, marker='o', linestyle='--')
                                                    plt.title('Elbow Method for KMeans')
                                                    plt.xlabel('Number of clusters')
                                                    plt.ylabel('Inertia')
                                                    plt.show()
                                                    st.pyplot(plt,use_container_width = True)

                    st.divider()

                    if best_model_clust == "KMeans":  
                        sample_silhouette_values = silhouette_samples(X, best_labels)

                        col1, col2 = st.columns((0.2,0.8))  
                        with col1:  
                            with st.container(): 
                        
                                silhouette_df = pd.DataFrame({'Data Point Index': np.arange(len(X)),'Cluster': best_labels,'Silhouette Coefficient': sample_silhouette_values})
                                st.dataframe(silhouette_df,hide_index=True, use_container_width=True)

                                with col2:  
                                    with st.container(): 

                                        y_lower = 10
                                        plt.figure(figsize=(8,3))

                                        for i in range(3):  
                                            ith_cluster_silhouette_values = sample_silhouette_values[best_labels == i]
                                            ith_cluster_silhouette_values.sort()
                                            size_cluster_i = ith_cluster_silhouette_values.shape[0]
                                            y_upper = y_lower + size_cluster_i

                                            color = plt.cm.nipy_spectral(float(i) / 3)
                                            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
                                            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                                            y_lower = y_upper + 10

                                        plt.axvline(x=silhouette_score(X, best_labels), color="red", linestyle="--")
                                        plt.title("Silhouette plot for the best model (KMeans)")
                                        plt.xlabel("Silhouette coefficient")
                                        plt.ylabel("Cluster")
                                        plt.show()
                                        st.pyplot(plt,use_container_width = True)        

#---------------------------------------------------------------------------------------------------------------------------------
            with tab7:
               
                if ml_type == 'Classification':        
                        
                        best_metrics=results_df.loc[results_df["Model"] == best_model_clf].iloc[0].to_dict()
                        final_results_df = pd.DataFrame({"Metric": ["Type of Problem",
                                                    "Target Variable",
                                                    "Type of Target",
                                                    "Scaling Method", 
                                                    "Feature Selection",
                                                    "Best Algorithm", 
                                                    "Accuracy", 
                                                    "AUC", 
                                                    "Precision", 
                                                    "Recall", 
                                                    "F1 Score", 
                                                    #"Best Feature(s)",
                                                    ],
                                            "Value": [ml_type,
                                                    target_variable,
                                                    target_type,
                                                    scaling_method, 
                                                    f_sel_method,
                                                    best_model_clf, 
                                                    round(best_metrics["Accuracy"],2), 
                                                    round(best_metrics["AUC"],2), 
                                                    round(best_metrics["Precision"],2),
                                                    round(best_metrics["Recall"],2), 
                                                    round(best_metrics["F1 Score"],2), 
                                                    #', '.join(best_features), 
                                                    ]})
                        col1, col2 = st.columns((0.2,0.8))
                        with col1:

                            st.subheader("Output",divider='blue')
                            #st.dataframe(final_results_df, hide_index=True, use_container_width=True)
                            st.table(final_results_df)

                        with col2:
                            X_test_results = X_test.copy()  
                            X_test_results["Actual"] = y_test
                            X_test_results["Predicted Label"] = y_pred_best
                            if y_proba_best is not None:
                                if target_type == "Binary":
                                    X_test_results["Prediction Score"] = y_proba_best  # For binary classification, use the second column of predict_proba
                                else:
                                    for i in range(y_proba_best.shape[1]):
                                        X_test_results[f"Class {i} Probability"] = y_proba_best[:, i]

                            st.subheader("Prediction & Score",divider='blue')
                            st.dataframe(X_test_results, use_container_width=True)
                            st.download_button(label="📥 Download predicted data as CSV",data=X_test_results.to_csv(index=False),file_name="classification_predictions.csv",mime="text/csv")

            #----------------------------------------  
                if ml_type == 'Regression':  

                        best_metrics=results_df.loc[results_df["Model"] == best_model_reg].iloc[0].to_dict()
                        final_results_df = pd.DataFrame({"Metric": ["Type of Problem",
                                                    "Target Variable",
                                                    "Scaling Method", 
                                                    "Feature Selection",
                                                    "Best Algorithm", 
                                                    "MAE", 
                                                    "MSE", 
                                                    "RMSE", 
                                                    "R2", 
                                                    "MAPE", 
                                                    #"Best Feature(s)",
                                                    ],
                                            "Value": [ml_type,
                                                    target_variable,
                                                    scaling_method, 
                                                    f_sel_method,
                                                    best_model_reg, 
                                                    round(best_metrics["MAE"],2), 
                                                    round(best_metrics["MSE"],2), 
                                                    round(best_metrics["RMSE"],2),
                                                    round(best_metrics["R2"],2), 
                                                    round(best_metrics["MAPE"],2), 
                                                    #', '.join(best_features), 
                                                    ]})
                        col1, col2 = st.columns((0.2,0.8))
                        with col1:

                            st.subheader("Output",divider='blue')                            
                            #st.dataframe(final_results_df, hide_index=True, use_container_width=True)
                            st.table(final_results_df)

                        with col2:
                             
                            best_model.fit(X_train, y_train)
                            y_pred_best = best_model.predict(X_test)
                            X_test_results_reg = X_test.copy()  
                            X_test_results_reg["Actual"] = y_test 
                            X_test_results_reg["Predicted"] = y_pred_best 

                            st.subheader("Prediction & Score",divider='blue')
                            st.dataframe(X_test_results_reg, use_container_width=True)
                            st.download_button(label="📥 Download predicted data as CSV",data=X_test_results_reg.to_csv(index=False),file_name="regression_predictions.csv",mime="text/csv")

            #----------------------------------------  
                if ml_type == 'Clustering':    
                     
                        best_metrics=results_df.loc[results_df["Algorithm"] == best_model_clust].iloc[0].to_dict()
                        final_results_df = pd.DataFrame({"Metric": ["Type of Problem",
                                                    "Target Variable",
                                                    "Scaling Method", 
                                                    "Feature Selection",
                                                    "Best Algorithm", 
                                                    "Silhouette", 
                                                    "Calinski-Harabasz", 
                                                    "Davies-Bouldin", 
                                                    "Homogeneity", 
                                                    "Rand Index", 
                                                    #"Best Feature(s)",
                                                    ],
                                            "Value": [ml_type,
                                                    target_variable,
                                                    scaling_method, 
                                                    f_sel_method,
                                                    best_model_clust, 
                                                    round(best_metrics["Silhouette"],2), 
                                                    round(best_metrics["Calinski-Harabasz"],2), 
                                                    round(best_metrics["Davies-Bouldin"],2),
                                                    round(best_metrics["Homogeneity"],2), 
                                                    round(best_metrics["Rand Index"],2), 
                                                    #', '.join(best_features), 
                                                    ]})
                        col1, col2 = st.columns((0.2,0.8))
                        with col1:

                            st.subheader("Output",divider='blue')                            
                            #st.dataframe(final_results_df, hide_index=True, use_container_width=True)
                            st.table(final_results_df)
