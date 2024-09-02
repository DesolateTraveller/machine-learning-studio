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
from sklearn.metrics import accuracy_score, auc, roc_auc_score, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
#----------------------------------------
# Model Validation

#----------------------------------------
#from pycaret.classification import setup, compare_models, pull, save_model, evaluate_model
#from pycaret.classification import setup, compare_models, predict_model, pull, plot_model, create_model, ensemble_model, blend_models, stack_models, tune_model, save_model
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
st.set_page_config(page_title="ML Studio | v0.1",
                   layout="wide",
                   page_icon="💻",              
                   initial_sidebar_state="auto")
#----------------------------------------
st.title(f""":rainbow[Machine Learning (ML) Studio]""")
st.markdown(
    '''
    Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a> ( :envelope: [Email](mailto:avijit.mba18@gmail.com) | :bust_in_silhouette: [LinkedIn](https://www.linkedin.com/in/avijit2403/) | :computer: [GitHub](https://github.com/DesolateTraveller) ) |
    for best view of the app, please **zoom-out** the browser to **75%**.
    ''',
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
models = {
    "Logistic Regression": LogisticRegression(),
    "Ridge Classifier": RidgeClassifier(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    #"CatBoost Classifier": CatBoostClassifier(verbose=0),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Ada Boost Classifier": AdaBoostClassifier(),
    "Extra Trees Classifier": ExtraTreesClassifier(),
    #"Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    #"Light Gradient Boosting Machine": LGBMClassifier(),
    "K Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    #"Extreme Gradient Boosting": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
    #"SVM - Linear Kernel": SVC(kernel="linear", probability=True)
    }
#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

st.sidebar.header("Input", divider='blue')
st.sidebar.info('Please choose from the following options to start the application.', icon="ℹ️")
ml_type = st.sidebar.selectbox("**:blue[Pick your Problem Type]**", ["None", "Classification", "Clustering", "Regression",])
                                                   
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
        st.sidebar.divider()
        if target_variable == "None":
            st.warning("Please choose a target variable to proceed with the analysis.")

#---------------------------------------------------------------------------------------------------------------------------------
        else:  
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["**Information**","**Visualizations**","**Cleaning**","**Transformation**","**Performance**","**Results**",])
            
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
                        
                        missing_values = check_missing_values(df)
                        if missing_values.empty:
                            st.success("**No missing values found!**")
                        else:
                            st.warning("**Missing values found!**")
                            st.write("**Number of missing values:**")
                            st.table(missing_values)

                            with col2:                 
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
                        treatment_option = st.selectbox("**:blue[Select a treatment option:]**", ["Cap Outliers","Drop Outliers", ])

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

                    st.sidebar.info(":blue-background[Feature Engineering]")
                     
                    col1, col2, col3= st.columns((0.25,0.5,0.25))  

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

                        scaling_method = st.sidebar.selectbox("**:blue[Choose a Scaling Method]**", ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
                        df = scale_features(df,scaling_method)
                        st.info("Data is scaled for further treatment")
                        #st.dataframe(df.head())

                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Download Scaled Data (for review)", data=csv, file_name='scaled_data.csv', mime='text/csv')

                    #----------------------------------------

                    with col2:   

                        st.subheader("Feature Selection",divider='blue')

                        f_sel_method = ['Method 1 : VIF', 
                                        'Method 2 : Selectkbest',
                                        'Method 3 : VarianceThreshold']
                        f_sel_method = st.sidebar.selectbox("**:blue[Choose a feature selection method]**", f_sel_method)
                        #st.divider()                    

                        if f_sel_method == 'Method 1 : VIF':

                            #st.subheader("Feature Selection (Method 1):",divider='blue')
                            st.markdown("**Method 1 : VIF**")
                            vif_threshold = st.number_input("**VIF Threshold**", 1.5, 10.0, 5.0)

                            st.markdown(f"Iterative VIF Thresholding (Threshold: {vif_threshold})")
                            #X = df.drop(columns = target_variable)
                            vif_data = drop_high_vif_variables(df, vif_threshold)
                            #vif_data = vif_data.drop(columns = target_variable)
                            selected_features = vif_data.columns
                            st.markdown("**Selected Features (considering VIF values in ascending orders)**")
                            st.write("No of features before feature-selection :",df.shape[1])
                            st.write("No of features after feature-selection :",len(selected_features))
                            st.table(selected_features)
                            #st.table(vif_data)

                        if f_sel_method == 'Method 2 : Selectkbest':

                            #st.subheader("Feature Selection (Method 2):",divider='blue')                        
                            st.markdown("**Method 2 : Selectkbest**")          
                            method = st.selectbox("**Select kBest Method**", ["f_classif", "f_regression", "chi2", "mutual_info_classif"])
                            num_features_to_select = st.slider("**Select Number of Independent Features**", min_value=1, max_value=len(df.columns), value=5)

                            if "f_classif" in method:
                                feature_selector = SelectKBest(score_func=f_classif, k=num_features_to_select)

                            elif "f_regression" in method:
                                feature_selector = SelectKBest(score_func=f_regression, k=num_features_to_select)

                            elif "chi2" in method:
                                # Make sure the data is non-negative for chi2
                                df[df < 0] = 0
                                feature_selector = SelectKBest(score_func=chi2, k=num_features_to_select)

                            elif "mutual_info_classif" in method:
                                # Make sure the data is non-negative for chi2
                                df[df < 0] = 0
                                feature_selector = SelectKBest(score_func=mutual_info_classif, k=num_features_to_select)

                            X = df.drop(columns = target_variable)  # Adjust 'Target' to your dependent variable
                            y = df[target_variable]  # Adjust 'Target' to your dependent variable
                            X_selected = feature_selector.fit_transform(X, y)

                            # Display selected features
                            selected_feature_indices = feature_selector.get_support(indices=True)
                            selected_features_kbest = X.columns[selected_feature_indices]
                            st.markdown("**Selected Features (considering values in 'recursive feature elimination' method)**")
                            st.write("No of features before feature-selection :",df.shape[1])
                            st.write("No of features after feature-selection :",len(selected_features))
                            st.table(selected_features_kbest)
                            selected_features = selected_features_kbest.copy()

                        if f_sel_method == 'Method 3 : VarianceThreshold':

                            st.markdown("**Method 3 : VarianceThreshold**")  
                            threshold = st.number_input("Variance Threshold", min_value=0.0, step=0.01, value=0.0)  

                            X = df.drop(columns = target_variable)  
                            y = df[target_variable]
                            selector = VarianceThreshold(threshold=threshold)
                            X_selected = selector.fit_transform(X)

                            selected_feature_indices = selector.get_support(indices=True)
                            selected_features_vth = X.columns[selected_feature_indices]          
                            st.markdown("**Selected Features (considering values in 'variance threshold' method)**") 
                            st.write("No of features before feature-selection :",df.shape[1])
                            st.write("No of features after feature-selection :",len(selected_features))                   
                            st.table(selected_features_vth)
                            selected_features = selected_features_vth.copy()

                    #----------------------------------------

                    with col3:                

                        st.subheader("Dataset Splitting Criteria",divider='blue')

                        train_size = st.slider("**Test Size (as %)**", 10, 90, 70, 5)
                        test_size = st.slider("**Test Size (as %)**", 10, 50, 30, 5)    
                        random_state = st.number_input("**Random State**", 0, 100, 42)
                        n_jobs = st.number_input("**Parallel Processing (n_jobs)**", -10, 10, 1)    

                        X = df.drop(columns = [target_variable])
                        y = df[target_variable]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

#---------------------------------------------------------------------------------------------------------------------------------
            with tab5:

                st.info("Please note that there may be some processing delay during the AutoML execution.")
                st.sidebar.divider()
                st.sidebar.info(f"**Selected Algorithm: {ml_type}**")
                 
                if ml_type == 'Classification': 

                    clf_typ = st.sidebar.selectbox("**:blue[Choose the type of target]**", ["Binary", "MultiClass"]) 
                    if clf_typ == 'Binary':
                        #if st.sidebar.button("Submit"):

                            col1, col2 = st.columns(2)  
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
                                        st.dataframe(results_df,hide_index=True, use_container_width=True)

                                        st.divider()

                                        best_model_acc = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
                                        st.write(f"The best model is (accuracy): **{best_model_acc}**")

                            with col2:

                                with st.container():  

                                    st.subheader("Graph",divider='blue')
                                    best_model = models[best_model_acc]
                                    best_model.fit(X_train, y_train)
                                    y_pred_best = best_model.predict(X_test)
                                    y_proba_best = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

                                    analysis_option = st.selectbox("Choose analysis type", ["Confusion Matrix", "AUC Curve"])

                                    if analysis_option == "Confusion Matrix":
                                        cm = confusion_matrix(y_test, y_pred_best)
                                        plt.figure(figsize=(10,3))
                                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                                        plt.title(f"Confusion Matrix for {best_model_acc}")
                                        plt.xlabel("Predicted")
                                        plt.ylabel("Actual")
                                        st.pyplot(plt,use_container_width=True)

                                    if analysis_option == "AUC Curve" and y_proba_best is not None:
                                        fpr, tpr, _ = roc_curve(y_test, y_proba_best)
                                        plt.figure(figsize=(10,3))
                                        plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {auc(fpr, tpr):.2f}")
                                        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
                                        plt.xlabel("False Positive Rate")
                                        plt.ylabel("True Positive Rate")
                                        plt.title(f"AUC Curve for {best_model_acc}")
                                        plt.legend(loc="lower right")
                                        st.pyplot(plt,use_container_width=True)


                            st.subheader("Importance",divider='blue')

                            importances = best_model.feature_importances_
                            indices = np.argsort(importances)[::-1]
                            plt.figure(figsize=(10,3))
                            plt.title(f"Feature Importances for {best_model_acc}")
                            plt.bar(range(X_train.shape[1]), importances[indices], align="center")
                            plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
                            plt.xlim([-1, X_train.shape[1]])
                            st.pyplot(plt,use_container_width=True)
