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
import shutil
import sweetviz as sv
import pygwalker as pyg
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
#st.info('**Disclaimer : :blue[Thank you for visiting the app] | Unauthorized uses or copying of the app is strictly prohibited | Click the :blue[sidebar] to follow the instructions to start the applications.**', icon="ℹ️")
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
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        df = pd.DataFrame()
    return df

@st.cache_data(ttl="2h")
def label_encode(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df

@st.cache_data(ttl="2h")
def onehot_encode(df, column):
    ohe = OneHotEncoder(sparse=False)
    encoded_cols = ohe.fit_transform(df[[column]])
    encoded_df = pd.DataFrame(encoded_cols, columns=[f"{column}_{cat}" for cat in ohe.categories_[0]])
    df = df.drop(column, axis=1).join(encoded_df)
    return df
#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

st.sidebar.header("Input", divider='blue')
st.sidebar.info('Please choose from the following options to start the application.', icon="ℹ️")
ml_type = st.sidebar.selectbox("**:blue[Pick your Problem Type]**", ["None", "Classification", "Clustering", "Image Classification","Regression"])
if ml_type == "None":
        st.warning("Please choose an algorithm in the sidebar to proceed with the analysis.")
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
        if target_variable == "None":
            st.warning("Please choose a target variable to proceed with the analysis.")

#---------------------------------------------------------------------------------------------------------------------------------
        else:  
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["**Information**",
                                             "**Visualizations**",
                                            "**Transformation**",
                                            "**Development**",
                                            "**Performance**",
                                            "**Importance**",])
            
#---------------------------------------------------------------------------------------------------------------------------------
            with tab1:

                #st.subheader("**Data Analysis**",divider='blue')
                col1, col2, col3, col4, col5, col6 = st.columns(6)

                col1.metric('**Number of input values (rows)**', df.shape[0], help='number of rows in the dataframe')
                col2.metric('**Number of variables (columns)**', df.shape[1], help='number of columns in the dataframe')     
                col3.metric('**Number of numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
                col4.metric('**Number of categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')
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
                
                    st.subheader("Missing Values Check & Treatment",divider='blue')
                    col1, col2 = st.columns((0.2,0.8))

                    with col1:
                        def check_missing_values(data):
                            missing_values = data.isnull().sum()
                            missing_values = missing_values[missing_values > 0]
                            return missing_values 
                        missing_values = check_missing_values(df)

                        if missing_values.empty:
                            st.success("**No missing values found!**")
                        else:
                            st.warning("**Missing values found!**")
                            st.write("**Number of missing values:**")
                            st.table(missing_values)

                            with col2:        
                                #treatment_option = st.selectbox("**Select a treatment option**:", ["Impute with Mean","Drop Missing Values", ])
        
                                # Perform treatment based on user selection
                                #if treatment_option == "Drop Missing Values":
                                    #df = df.dropna()
                                    #st.success("Missing values dropped. Preview of the cleaned dataset:")
                                    #st.table(df.head())
            
                                #elif treatment_option == "Impute with Mean":
                                    #df = df.fillna(df.mean())
                                    #st.success("Missing values imputed with mean. Preview of the imputed dataset:")
                                    #st.table(df.head())
                                 
                                # Function to handle missing values for numerical variables
                                def handle_numerical_missing_values(data, numerical_strategy):
                                    imputer = SimpleImputer(strategy=numerical_strategy)
                                    numerical_features = data.select_dtypes(include=['number']).columns
                                    data[numerical_features] = imputer.fit_transform(data[numerical_features])
                                    return data

                                # Function to handle missing values for categorical variables
                                def handle_categorical_missing_values(data, categorical_strategy):
                                    imputer = SimpleImputer(strategy=categorical_strategy, fill_value='no_info')
                                    categorical_features = data.select_dtypes(exclude=['number']).columns
                                    data[categorical_features] = imputer.fit_transform(data[categorical_features])
                                    return data            

                                numerical_strategies = ['mean', 'median', 'most_frequent']
                                categorical_strategies = ['constant','most_frequent']
                                st.write("**Missing Values Treatment:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    selected_numerical_strategy = st.selectbox("**Select a strategy for treatment : Numerical variables**", numerical_strategies)
                                with col2:
                                    selected_categorical_strategy = st.selectbox("**Select a strategy for treatment : Categorical variables**", categorical_strategies)  
                                
                                #if st.button("**Apply Missing Values Treatment**"):
                                cleaned_df = handle_numerical_missing_values(df, selected_numerical_strategy)
                                cleaned_df = handle_categorical_missing_values(cleaned_df, selected_categorical_strategy)   
                                st.table(cleaned_df.head(2))

                                # Download link for treated data
                                st.download_button("**Download Treated Data**", cleaned_df.to_csv(index=False), file_name="treated_data.csv")

                    #with col2:

                    st.subheader("Duplicate Values Check",divider='blue') 
                    if st.checkbox("Show Duplicate Values"):
                        if missing_values.empty:
                            st.table(df[df.duplicated()].head(2))
                        else:
                            st.table(cleaned_df[cleaned_df.duplicated()].head(2))

                    #with col4:

                        #x_column = st.selectbox("Select x-axis column:", options = df.columns.tolist()[0:], index = 0)
                        #y_column = st.selectbox("Select y-axis column:", options = df.columns.tolist()[0:], index = 1)
                        #chart = alt.Chart(df).mark_boxplot(extent='min-max').encode(x=x_column,y=y_column)
                        #st.altair_chart(chart, theme=None, use_container_width=True)  

                    st.subheader("Outliers Check & Treatment",divider='blue')
                    def check_outliers(data):
                                # Assuming we're checking for outliers in numerical columns
                                numerical_columns = data.select_dtypes(include=[np.number]).columns
                                outliers = pd.DataFrame(columns=['Column', 'Number of Outliers'])

                                for column in numerical_columns:
                                    Q1 = data[column].quantile(0.25)
                                    Q3 = data[column].quantile(0.75)
                                    IQR = Q3 - Q1

                                    # Define a threshold for outliers
                                    threshold = 1.5

                                    # Find indices of outliers
                                    outliers_indices = ((data[column] < Q1 - threshold * IQR) | (data[column] > Q3 + threshold * IQR))

                                    # Count the number of outliers
                                    num_outliers = outliers_indices.sum()
                                    outliers = outliers._append({'Column': column, 'Number of Outliers': num_outliers}, ignore_index=True)

                                return outliers
                
                    if missing_values.empty:
                        df = df.copy()
                    else:
                        df = cleaned_df.copy()

                    col1, col2 = st.columns((0.2,0.8))

                    with col1:
                        # Check for outliers
                        outliers = check_outliers(df)

                        # Display results
                        if outliers.empty:
                            st.success("No outliers found!")
                        else:
                            st.warning("**Outliers found!**")
                            st.write("**Number of outliers:**")
                            st.table(outliers)
                    
                    with col2:
                        # Treatment options
                        treatment_option = st.selectbox("**Select a treatment option:**", ["Cap Outliers","Drop Outliers", ])

                            # Perform treatment based on user selection
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

                                    # Cap outliers
                                    df[column] = np.where(df[column] < Q1 - threshold * IQR, Q1 - threshold * IQR, df[column])
                                    df[column] = np.where(df[column] > Q3 + threshold * IQR, Q3 + threshold * IQR, df[column])

                                    st.success("Outliers capped. Preview of the capped dataset:")
                                    st.write(df.head())

#---------------------------------------------------------------------------------------------------------------------------------
                with tab4:
                     
                    col1, col2, col3, col4 = st.columns(4)  

                    with col1:
                         
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

