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
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyoff
import altair as alt
#import dabl
#----------------------------------------
import traceback
import shutil
import sweetviz as sv
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer
#----------------------------------------
# Model Building
import xgboost as xgb
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
#
from sklearn.ensemble import GradientBoostingClassifier
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
#----------------------------------------
# Model Validation

#----------------------------------------
#from pycaret.classification import setup, compare_models, pull, save_model, evaluate_model

#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
st.set_page_config(page_title="ML Studio",
                   layout="wide",
                   #page_icon=               
                   initial_sidebar_state="auto")
#----------------------------------------
st.title(f""":rainbow[Machine Learning (ML) Studio | v0.1]""")
st.markdown('Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a>', 
            unsafe_allow_html=True)
st.info('**Disclaimer : :blue[Thank you for visiting the app] | Unauthorized uses or copying of the app is strictly prohibited | Follow the instructions to start the applications.**', icon="ℹ️")
#----------------------------------------
# Set the background image
st.divider()

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

@st.cache_data(ttl="2h")
def load_file(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(file, sep=None, engine='python', encoding='utf-8', parse_dates=True, infer_datetime_format=True)
        st.session_state['dataframe'] = df
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file)
        st.session_state['dataframe'] = df
    else:
        st.error("Unsupported file format")
        df = pd.DataFrame()
    return df


#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

file = st.file_uploader("**:blue[Choose a file]**", type=["csv", "xls", "xlsx"], accept_multiple_files=False, key="file_upload")
if file is not None:
    df = load_file(file)
    stats_expander = st.expander("**Preview of Data**", expanded=True)
    with stats_expander:  
        st.table(df.head(2))

#---------------------------------------------------------------------------------------------------------------------------------     
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Information**","**Visualization**","**Build**","**Development**","**Performance**",])
    with tab1:

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric('**Number of input values (rows)**', df.shape[0], help='number of rows in the dataframe')
        col2.metric('**Number of variables (columns)**', df.shape[1], help='number of columns in the dataframe')     
        col3.metric('**Number of numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
        col4.metric('**Number of categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')
        #st.divider()           

        stats_expander = st.expander("**Exploratory Data Analysis (EDA)**", expanded=False)
        with stats_expander:        
            st.table(df.head(2))

#---------------------------------------------------------------------------------------------------------------------------------
    with tab2:

        plot_option = st.selectbox("**Choose Plot**", ["Line Chart", "Histogram", "Scatter Plot", "Bar Chart", "Box Plot"])


#---------------------------------------------------------------------------------------------------------------------------------
    with tab3:

        col1,col2, col3 = st.columns([0.2,0.3,0.5])
        with col1:

            task = st.selectbox("**Select ML task**", ["Classification", "Regression", "Clustering", "Anomaly Detection", "Time Series Forecasting"])

        with col2:
                
            stats_expander = st.expander("**Select Columns**", expanded=False)
            with stats_expander:

                target_column = st.selectbox("Select target column", df.columns) if task in ["Classification", "Regression", "Time Series Forecasting"] else None
                numerical_columns = st.multiselect("Select numerical columns", df.columns)
                categorical_columns = st.multiselect("Select categorical columns", df.columns)

        with col3:
                
            stats_expander = st.expander("**Tune Parameters**", expanded=False)
            with stats_expander:

                # Data Preparation
                handle_missing_data = st.toggle("Handle Missing Data", value=True)
                handle_outliers = st.toggle("Handle Outliers", value=True)
        
                # Scale and Transform
                normalize = st.checkbox("Normalize", value=False)
                normalize_method = st.selectbox("Normalize Method", ["zscore", "minmax", "maxabs", "robust"], index=0 if normalize else -1) if normalize else None
                transformation = st.checkbox("Apply Transformation", value=False)
                transformation_method = st.selectbox("Transformation Method", ["yeo-johnson", "quantile"], index=0 if transformation else -1) if transformation else None
        
                # Feature Engineering
                polynomial_features = st.checkbox("Polynomial Features", value=False)
                polynomial_degree = st.slider("Polynomial Degree", 2, 5, 2) if polynomial_features else None
        
                # Feature Selection
                remove_multicollinearity = st.checkbox("Remove Multicollinearity", value=False)
                multicollinearity_threshold = st.slider("Multicollinearity Threshold", 0.5, 1.0, 0.9) if remove_multicollinearity else None
        
                if not (task == "Anomaly Detection" or task == "Clustering") :
                    feature_selection = st.checkbox("Feature Selection", value=False)
                    feature_selection_method = st.selectbox("Feature Selection Method", ["classic", "exhaustive"], index=0 if feature_selection else -1) if feature_selection else None
                else:
                    feature_selection = None
                    feature_selection_method = None

