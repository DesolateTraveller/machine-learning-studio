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
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, save_model as cls_save, pull as cls_pull, plot_model as cls_plot
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, save_model as reg_save, pull as reg_pull, plot_model as reg_plot
from pycaret.clustering import setup as clu_setup, create_model as clu_create, plot_model as clu_plot, save_model as clu_save, pull as clu_pull
from pycaret.anomaly import setup as ano_setup, create_model as ano_create, plot_model as ano_plot, save_model as ano_save, pull as ano_pull
from pycaret.time_series import setup as ts_setup, compare_models as ts_compare, save_model as ts_save, pull as ts_pull, plot_model as ts_plot
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

@st.cache_data(ttl="2h")
def pywalkr(dataset):
    try:
        pyg_app = StreamlitRenderer(dataset)
        pyg_app.explorer()
    except Exception as e:
        st.error(str(e))

@st.cache_data(ttl="2h")
def update_progress(progress_bar, step, max_steps):
    progress = int((step / max_steps) * 100)
    t = f"Processing....Step {step}/{max_steps}"
    if step == max_steps:
        t="Process Completed"
    progress_bar.progress(progress, text=t)

@st.cache_data(ttl="2h")
def build_model(task):
    try:
        # Setup arguments for PyCaret
        setup_kwargs = {
                'data': df[numerical_columns + categorical_columns + ([target_column] if target_column else [])],
                'categorical_features': categorical_columns,
                'numeric_features': numerical_columns,
                'target': target_column,
                'preprocess': handle_missing_data,
                'remove_outliers': handle_outliers,
                'normalize': normalize,
                'normalize_method': normalize_method,
                'transformation': transformation,
                'transformation_method': transformation_method,
                'polynomial_features': polynomial_features,
                'polynomial_degree': polynomial_degree,
                'remove_multicollinearity': remove_multicollinearity,
                'multicollinearity_threshold': multicollinearity_threshold,
                'feature_selection': feature_selection,
                'feature_selection_method': feature_selection_method}
        
        pb = st.progress(0, text="Building Model...")
        if task == "Classification" and st.button("Run Classification"):
                
            df[target_column] = df[target_column].astype('category')
            df.dropna(subset=[target_column] + numerical_columns + categorical_columns, inplace=True)
                
            if len(df) < 2:
                st.error("Not enough data to split into train and test sets.")
                return
            
            update_progress(pb,1,7)
            exp = cls_setup(**setup_kwargs)
            update_progress(pb,2,7)
            best_model = cls_compare()
            update_progress(pb,3,7)
            st.dataframe(cls_pull())
            update_progress(pb,4,7)
            cls_plot(best_model, plot='auc',display_format="streamlit")
            cls_plot(best_model, plot='confusion_matrix',display_format="streamlit")
            update_progress(pb,5,7)
            st.image(cls_plot(best_model, plot='pr',save=True))
            update_progress(pb,6,7)
            cls_save(best_model, 'best_classification_model')
            st.write('Best Model based on metrics - ')
            st.write(best_model)
            update_progress(pb,7,7)

        elif task == "Regression" and st.button("Run Regression"):
            
            update_progress(pb,1,7)
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
            update_progress(pb,2,7)
            df.dropna(subset=[target_column] + numerical_columns + categorical_columns, inplace=True)
            update_progress(pb,3,7)                
            if len(df) < 2:
                st.error("Not enough data to split into train and test sets.")
                return
                
            exp = reg_setup(**setup_kwargs)
            best_model = reg_compare()
            update_progress(pb,4,7)
            st.dataframe(reg_pull())
            update_progress(pb,5,7)
            st.image(reg_plot(best_model, plot='residuals', save=True))
            st.image(reg_plot(best_model, plot='error', save=True))
            st.image(reg_plot(best_model, plot='error', save=True))
            update_progress(pb,6,7)
            reg_save(best_model, 'best_regression_model')
            st.write('Best Model based on metrics - ')
            st.write(best_model)
            update_progress(pb,7,7)
            
        elif task == "Clustering" and st.button("Run Clustering"):
            
            update_progress(pb,1,7)
            df.dropna(subset=numerical_columns + categorical_columns, inplace=True)
            update_progress(pb,2,7)
            setup_kwargs.pop('target')
            setup_kwargs.pop('feature_selection')
            setup_kwargs.pop('feature_selection_method')  
            update_progress(pb,3,7)
            exp = clu_setup(**setup_kwargs)
            best_model = clu_create('kmeans')
            update_progress(pb,4,7)
            clu_plot(best_model, plot='cluster', display_format='streamlit')
            clu_plot(best_model, plot='elbow', display_format='streamlit')
            update_progress(pb,5,7)
            st.write(best_model)
            st.dataframe(clu_pull())
            update_progress(pb,6,7)
            clu_save(best_model, 'best_clustering_model')
            st.write('Best Model based on metrics - ')
            st.write(best_model)
            update_progress(pb,7,7)

        elif task == "Anomaly Detection" and st.button("Run Anomaly Detection"):
                
            update_progress(pb,1,7)
            df.dropna(subset=numerical_columns + categorical_columns, inplace=True)
            update_progress(pb,2,7)
            setup_kwargs.pop('target')
            setup_kwargs.pop('feature_selection')
            setup_kwargs.pop('feature_selection_method')        
            update_progress(pb,3,7)
            exp = ano_setup(**setup_kwargs)
            best_model = ano_create('iforest')
            update_progress(pb,4,7)
            ano_plot(best_model, plot='tsne', display_format='streamlit')
            update_progress(pb,5,7)                
            st.write(best_model)
            st.dataframe(ano_pull())
            update_progress(pb,6,7)
            ano_save(best_model, 'best_anomaly_model')
            st.write('Best Model based on metrics - ')
            st.write(best_model)
            update_progress(pb,7,7)

        elif task == "Time Series Forecasting" :
                
            date_column = st.selectbox("Select date column", df.columns)
            if st.button("Run Time Series Forecasting"):
                update_progress(pb,1,5)
                df[date_column] = pd.to_datetime(df[date_column])
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                df.dropna(subset=[target_column], inplace=True)
                update_progress(pb,2,5)                
                df = df.set_index(date_column).asfreq('D')
                exp = ts_setup(df, target=target_column, numeric_imputation_target='mean', numeric_imputation_exogenous='mean')
                best_model = ts_compare()
                update_progress(pb,3,5)                    
                st.dataframe(ts_pull())
                ts_plot(best_model, plot='forecast', display_format="streamlit")
                ts_save(best_model, 'best_timeseries_model')
                update_progress(pb,4,5)
                st.write('Best Model based on metrics - ')
                st.write(best_model)
                update_progress(pb,5,5)

    except Exception as e:
            st.info("Please upload a file to start the EDA process.")
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Information**","**Visualization**","**Development**","**Performance**","**Importance**",])
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

        build_model(task)
