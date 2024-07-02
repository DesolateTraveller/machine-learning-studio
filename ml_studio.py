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
import sweetviz as sv
import pygwalker as pyg
#----------------------------------------
# Model Building
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
#
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
#import optuna.integration.lightgbm as lgb
from sklearn.metrics import classification_report,confusion_matrix
#----------------------------------------
# Model Performance
import shap
from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, f_regression, chi2, VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
#----------------------------------------
# Model Validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
#----------------------------------------
#from pycaret.classification import setup, compare_models, pull, save_model, evaluate_model
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
st.set_page_config(page_title="ML Studio",
                   layout="wide",
                   #page_icon=               
                   initial_sidebar_state="collapsed")
#----------------------------------------
st.title(f""":rainbow[Machine Learning (ML) Studio | v0.1]""")
st.markdown('Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a>', 
            unsafe_allow_html=True)
#t.info('**Disclaimer : :blue[Thank you for visiting the app] | Unauthorized uses or copying of the app is strictly prohibited | Click the :blue[sidebar] to follow the instructions to start the applications.**', icon="ℹ️")
#----------------------------------------
# Set the background image
st.divider()

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns((0.2,0.4,0.3,0.1))

with col1:
    problem_type = st.selectbox("Pick your Problem Type", ["Regression", "Classification", "Clustering", "Image Classification"])

with col2:
    if problem_type == "Regression":
        state = 1
    elif problem_type == "Classification":
        state = 2
    elif problem_type == "Clustering":
        state = 3
    else:
        state = 4

    if state == 4:
        img_zip_file = st.file_uploader("Upload your Dataset", type=['zip'])
    else:
        dataset_file = st.file_uploader("Upload your Dataset", type=['csv'])
    
with col3:
    if state != 4:
        if dataset_file:
            df = pd.read_csv(dataset_file)
            if state == 1 or state == 2:
                target_y = st.multiselect("**2.1 Target (Dependent) Variable**", df.columns)

with col4:
    train_btn = st.button("Train")

#----------------------------------------
stats_expander = st.expander("**Input Information**", expanded=False)
with stats_expander:  
    st.dataframe(df)
