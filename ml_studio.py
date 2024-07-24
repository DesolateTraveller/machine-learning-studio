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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
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

#----------------------------------------
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,MobileNetV2,ResNet50
from tensorflow.keras.models import Model, load_model
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

@st.cache_data(ttl="2h")
def __init__(self,dataframe):
        self.df = dataframe
        self.X = None
        self.y = None

@st.cache_data(ttl="2h")
def clean_and_scale_dataset(self):
        self.df = self.df.dropna()
        self.X_pre = self.df.drop(columns = self.df.columns[-1])
        self.y = self.df.iloc[:,-1]

        columns = self.X_pre.columns
        categorical_cols = self.X_pre.select_dtypes(include = ['object']).columns
        self.X = self.X_pre.copy()
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        for col in categorical_cols:
            temp_df = pd.DataFrame(self.X[col])
            encoded_col = encoder.fit_transform(temp_df)
            encoded_col_names = encoder.get_feature_names_out([col])
            self.X = self.X.drop(col, axis=1)
            self.X = pd.concat([self.X, pd.DataFrame(encoded_col, columns=encoded_col_names)], axis=1)
        
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        return self.X,self.y
    
####################### Classifiers ################################
@st.cache_data(ttl="2h")
def SVM(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = svm.SVC()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics

@st.cache_data(ttl="2h")    
def logistic_regression(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier =  LogisticRegression()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics

@st.cache_data(ttl="2h")
def dtree_classifier(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics

@st.cache_data(ttl="2h")    
def knn_classifier(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = KNeighborsClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics

@st.cache_data(ttl="2h")    
def naivebayes(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = GaussianNB()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics

@st.cache_data(ttl="2h")    
def adaboost(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = AdaBoostClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics

@st.cache_data(ttl="2h")    
def mlp(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = MLPClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics

@st.cache_data(ttl="2h")    
def gradient_boost(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = GradientBoostingClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics

@st.cache_data(ttl="2h")    
def random_forest_classifier(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = RandomForestClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics


##########################  Regressors ##################################

@st.cache_data(ttl="2h")
def linear_regression(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = LinearRegression()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics

@st.cache_data(ttl="2h")    
def dtree_regressor(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = tree.DecisionTreeRegressor()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics

@st.cache_data(ttl="2h")    
def SVR(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = svm.SVR()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics

@st.cache_data(ttl="2h")    
def ridge_regression(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = Ridge()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics

@st.cache_data(ttl="2h")    
def lasso(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = Lasso()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics

@st.cache_data(ttl="2h")    
def elasticnet(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = ElasticNet()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics

@st.cache_data(ttl="2h")    
def random_forest_regressor(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = RandomForestRegressor()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics

@st.cache_data(ttl="2h")    
def mlp_regressor(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = MLPRegressor()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics

@st.cache_data(ttl="2h")    
def knn_regressor(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = KNeighborsRegressor()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics

@st.cache_data(ttl="2h")    
def gradient_boost_regressor(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = GradientBoostingRegressor()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics
    
########################## Clustering ####################################

@st.cache_data(ttl="2h")
def kmeans(self,num_clusters = 5):
        X_scaled, _ = self.clean_and_scale_dataset()
        sil_scores = []

        if num_clusters == -1:
            for i in range(2, 11):
                kmeans = KMeans(n_clusters=i, random_state=42)
                labels = kmeans.fit_predict(X_scaled)
                sil_score = silhouette_score(X_scaled, labels)
                sil_scores.append(sil_score)
            num_clusters = np.argmax(sil_scores) + 2
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
        else:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled,labels)
        db_score = davies_bouldin_score(X_scaled,labels)
        metrics = {
            "Silhouette score":sil_score,
            "Davies Bouldin score":db_score
        }
        return kmeans, labels, metrics, num_clusters

@st.cache_data(ttl="2h")    
def agglomerative_clustering(self,num_clusters = 5):
        X_scaled, _ = self.clean_and_scale_dataset()
        sil_scores = []

        if num_clusters == -1:
            for i in range(2, 11):
                agg = AgglomerativeClustering(n_clusters=i)
                labels = agg.fit_predict(X_scaled)
                sil_score = silhouette_score(X_scaled, labels)
                sil_scores.append(sil_score)
            num_clusters = np.argmax(sil_scores) + 2
            agg = AgglomerativeClustering(n_clusters=num_clusters)
            labels = agg.fit_predict(X_scaled)
        else:
            agg = AgglomerativeClustering(n_clusters=num_clusters)
            labels = agg.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled,labels)
        db_score = davies_bouldin_score(X_scaled,labels)
        metrics = {
            "Silhouette score":sil_score,
            "Davies Bouldin score":db_score
        }
        return agg, labels, metrics, num_clusters

@st.cache_data(ttl="2h")    
def dbscan(self,eps=2, min_samples = 5):
        X_scaled, _ = self.clean_and_scale_dataset()
        dbscan = DBSCAN(eps = eps,min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled,labels)
        db_score = davies_bouldin_score(X_scaled,labels)
        metrics = {
            "Silhouette score":sil_score,
            "Davies Bouldin score":db_score
        }
        return dbscan, labels, metrics

    # def predict(self,model,x):
    #     answer = model.predict(x)
    #     return answer
    
@st.cache_data(ttl="2h")
def pick_best_classifier(self):
        model_name,model,opt_score = "",None,0.0

        classifiers = {
            "SVM":self.SVM(),
            "Logistic Regression":self.logistic_regression(),
            "Decision Tree": self.dtree_classifier(),
            "KNN":self.knn_classifier(),
            "Naive Bayes": self.naivebayes(),
            "Ada boost": self.adaboost(),
            "Gradient boost": self.gradient_boost(),
            "Random Forest": self.random_forest_classifier(),
        }      
        
        for name,tup in classifiers.items():
            clf,score = tup
            # print(score['Accuracy'])
            if(score['Accuracy'] > opt_score):
                opt_score = score['Accuracy']
                model = clf
                model_name = name

        return classifiers, model_name, model, opt_score

@st.cache_data(ttl="2h")    
def pick_best_regressor(self):
        model_name, model, opt_score = "",None,0

        regressors = {
            "Linear Regression": self.linear_regression(),
            "Decision tree regression": self.dtree_regressor(),
            "Support vector regression": self.SVR(),
            "Ridge regression":self.ridge_regression(),
            "Lasso regression":self.lasso(),
            "ElasticNet regression":self.elasticnet(),
            "Random Forest regression":self.random_forest_regressor(),
            "Multi-layer perceptron":self.mlp_regressor(),
            "KNN":self.knn_regressor(),
            "Gradient Boosting regression":self.gradient_boost_regressor()
        }

        for name,tup in regressors.items():
            clf,score = tup
            if(score['R2 score'] > opt_score):
                opt_score = score['R2 score']
                model = clf
                model_name = name

        return regressors, model_name, model, opt_score

########################## Visualization####################################

@st.cache_data(ttl="2h") 
def threshold_regression(r2_score, threshold=0.7):
    if r2_score >= threshold:
        suggestion = f"The R2 score ({r2_score:.2f}) indicates a good fit for the regression model. No further action is required."
    else:
        if r2_score >= 0.6:
            suggestion = f"The R2 score ({r2_score:.2f}) suggests that the regression model may need some adjustments. Consider the following:\n\n"
            suggestion += "- Adding additional relevant features to the model.\n"
            suggestion += "- Fine-tuning hyperparameters such as regularization strength or learning rate.\n"
            suggestion += "- Checking for multicollinearity among features and addressing it if present.\n"
        elif r2_score >= 0.5:
            suggestion = f"The R2 score ({r2_score:.2f}) indicates a moderate fit for the regression model. To improve the performance, you may:\n\n"
            suggestion += "- Experiment with different algorithms or ensemble methods.\n"
            suggestion += "- Feature engineering to create new informative features.\n"
            suggestion += "- Addressing outliers or anomalies in the data.\n"
        else:
            suggestion = f"The R2 score ({r2_score:.2f}) indicates a poor fit for the regression model. Consider the following steps to enhance the model performance:\n\n"
            suggestion += "- Exploring more complex models or nonlinear transformations of features.\n"
            suggestion += "- Collecting additional relevant data to improve model generalization.\n"
            suggestion += "- Conducting thorough feature selection to retain only the most informative features.\n"

    return suggestion

@st.cache_data(ttl="2h") 
def threshold_classification(accuracy, threshold=0.7):
    if accuracy >= threshold:
        suggestion = f"The accuracy score ({accuracy:.2f}) indicates a good performance for the classification model. No further action is required."
    else:
        if accuracy >= 0.6:
            suggestion = f"The accuracy score ({accuracy:.2f}) suggests that the classification model may need some adjustments. Consider the following:\n\n"
            suggestion += "- Adding more data or augmenting existing data to improve model generalization.\n"
            suggestion += "- Experimenting with different algorithms or hyperparameters.\n"
            suggestion += "- Performing feature engineering to create more informative features.\n"
        elif accuracy >= 0.5:
            suggestion = f"The accuracy score ({accuracy:.2f}) indicates a moderate performance for the classification model. To improve the accuracy, you may:\n\n"
            suggestion += "- Fine-tune the model's hyperparameters using techniques such as grid search or random search.\n"
            suggestion += "- Conducting thorough feature selection to retain only the most relevant features.\n"
            suggestion += "- Handling class imbalance issues using techniques such as oversampling or undersampling.\n"
        else:
            suggestion = f"The accuracy score ({accuracy:.2f}) suggests a poor performance for the classification model. Consider the following steps to enhance the model performance:\n\n"
            suggestion += "- Collecting more labeled data to improve model training.\n"
            suggestion += "- Exploring different machine learning algorithms or ensemble methods.\n"
            suggestion += "- Addressing data preprocessing issues such as feature scaling or normalization.\n"

    return suggestion

@st.cache_data(ttl="2h") 
def threshold_clustering(silhouette_scores, threshold=0.5):
    suggestions = []
    for silhouette_score in silhouette_scores:
        if silhouette_score >= threshold:
            suggestion = f"The silhouette score ({silhouette_score:.2f}) indicates good cluster separation."
        else:
            suggestion = f"The silhouette score ({silhouette_score:.2f}) suggests poor cluster separation. Consider the following:\n\n"
            suggestion += "- Experimenting with different clustering algorithms.\n"
            suggestion += "- Tuning the hyperparameters of the chosen algorithm.\n"
            suggestion += "- Evaluating feature selection and engineering methods.\n"
        suggestions.append(suggestion)
    return suggestions

@st.cache_data(ttl="2h") 
def main(state, comp_table=None, X=None, y=None, df=None,model_names=None,models = None):
    if state == 1:  # Regression task

        array_cols = comp_table.dtypes[comp_table.dtypes == 'object'].index

        for col in array_cols:
            if comp_table[col].apply(lambda x: isinstance(x, np.ndarray)).any():
                comp_table = comp_table.drop(col, axis=1)
        st.subheader("Bar Graph")
        st.bar_chart(comp_table)
        st.subheader("Line Graph")
        st.line_chart(comp_table)
        st.subheader("Area Graph")
        st.area_chart(comp_table)
        st.subheader("Heatmap")
        st.set_option('deprecation.showPyplotGlobalUse', False)

        non_numeric_cols = comp_table.select_dtypes(include=['object']).columns
        for col in non_numeric_cols:
            comp_table[col] = pd.to_numeric(comp_table[col], errors='coerce')

        comp_table = comp_table.replace([np.inf, -np.inf], np.nan)
        sns.heatmap(comp_table, annot=True)
        st.pyplot()
        
        st.sidebar.subheader("Pick the Model You Want Suggestions")
        selected_model = st.sidebar.selectbox("Select Model", ["Best Model"] + model_names)

        if selected_model != "Best Model":
            r2_score = comp_table.loc[selected_model]["R2 score"]
            st.subheader(f"{selected_model} Model")
            st.write("R2 Score:", r2_score)
            suggestion = threshold_regression(r2_score)
            st.subheader("Model Performance Analysis")
            st.write(suggestion)
        else:
            max_index = comp_table["R2 score"].idxmax()
            r2_score = comp_table.loc[max_index]["R2 score"]
            st.subheader(f"{max_index} Model")
            st.write("This is the Best Model based on R2 Score")
            st.write("R2 Score:", r2_score)

    elif state == 2:  # Classification task

        st.header("Classification Visualizations")

        st.subheader("Metrics Comparison")

        colors = px.colors.qualitative.Plotly

        """ fig = px.bar(comp_table, x=comp_table.index, y=["Accuracy", "Precision", "Recall", "F1-score"], barmode="group", color_discrete_sequence=colors)
        fig.update_layout(title="Classification Metrics", xaxis_title="Model Name", yaxis_title="Metric Value")
        st.plotly_chart(fig)

        # Precision-Recall curve
        st.subheader("Precision-Recall Curve")
        fig = px.area(comp_table, x="Recall", y="Precision", color=comp_table.index, line_group=comp_table.index, color_discrete_sequence=colors)
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig) """

        array_cols = comp_table.dtypes[comp_table.dtypes == 'object'].index

        for col in array_cols:
            if comp_table[col].apply(lambda x: isinstance(x, np.ndarray)).any():
                comp_table = comp_table.drop(col, axis=1)

        st.subheader("Bar Graph")
        st.bar_chart(comp_table)
        st.subheader("Line Graph")
        st.line_chart(comp_table)
        st.subheader("Area Graph")
        st.area_chart(comp_table)
        st.subheader("Heatmap")
        st.set_option('deprecation.showPyplotGlobalUse', False)

        non_numeric_cols = comp_table.select_dtypes(include=['object']).columns
        for col in non_numeric_cols:
            comp_table[col] = pd.to_numeric(comp_table[col], errors='coerce')

        comp_table = comp_table.replace([np.inf, -np.inf], np.nan)

        sns.heatmap(comp_table, annot=True)
        st.pyplot()

        st.sidebar.subheader("Pick the Model You Want Suggestions")
        selected_model = st.sidebar.selectbox("Select Model", ["Best Model"] + model_names)

        if selected_model != "Best Model":
            accuracy = comp_table.loc[selected_model]["Accuracy"]
            st.subheader(f"{selected_model} Model")
            st.write("Accuracy:", accuracy)
            suggestion = threshold_classification(accuracy)
            st.subheader("Model Performance Analysis")
            st.write(suggestion)
        else:
            comp_table["Accuracy"] = pd.to_numeric(comp_table["Accuracy"], errors='coerce')

            comp_table["Accuracy"] = comp_table["Accuracy"].replace([np.inf, -np.inf], np.nan)

            max_index = comp_table["Accuracy"].idxmax()
            accuracy = comp_table.loc[max_index]["Accuracy"]
            st.subheader(f"{max_index} Model")
            st.write("This is the Best Model based on Accuracy")
            st.write("Accuracy:", accuracy)

    elif state == 3:  # Clustering task
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Elbow Method for Optimal Number of Clusters")
        nan_values = df.isna().sum().sum()

        if nan_values > 0:
            st.write("Warning: NaN values found in the data. Imputing NaN values with column means before proceeding.")
            df.fillna(df.mean(), inplace=True)

            
        distortions = []
        max_clusters = 10
        cluster_range = range(1, max_clusters + 1)

        for num_clusters in cluster_range:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(df)
            distortions.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, distortions, marker='o', linestyle='-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Distortion')
        plt.title('Elbow Method for KMeans Clustering')
        plt.xticks(cluster_range)
        st.pyplot()

        st.subheader("Silhouette Score for KMeans Clustering")
        silhouette_scores = []
        for num_clusters in cluster_range:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(df)
            if len(np.unique(cluster_labels)) < 2:
                silhouette_scores.append(np.nan)  # Skip silhouette score calculation
            else:
                silhouette_avg = silhouette_score(df, cluster_labels)
                silhouette_scores.append(silhouette_avg)

        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for KMeans Clustering')
        plt.xticks(cluster_range)
        st.pyplot()

        st.subheader("Threshold Analysis for KMeans Clustering")
        suggestions = threshold_clustering(silhouette_scores)
        for k, suggestion in enumerate(suggestions, start=2):
            st.write(suggestion)

        st.write("Agglomerative Clustering is hierarchical and does not typically use the elbow method.")
        st.write("You may visualize the resulting dendrogram instead.")
        
        agglomerative = AgglomerativeClustering().fit(df)
        plt.figure(figsize=(10, 6))
        dendrogram = shc.dendrogram(shc.linkage(df, method='ward'))
        plt.title('Dendrogram for Agglomerative Clustering')
        st.pyplot()

        st.subheader("Silhouette Score for Agglomerative Clustering")
        silhouette_scores = []
        max_clusters = 10
        cluster_range = range(2, max_clusters + 1)
        for num_clusters in cluster_range:
            agg = AgglomerativeClustering(n_clusters=num_clusters)
            cluster_labels = agg.fit_predict(df)
            if len(np.unique(cluster_labels)) < 2:
                silhouette_scores.append(np.nan)  # Skip silhouette score calculation
            else:
                silhouette_avg = silhouette_score(df, cluster_labels)
                silhouette_scores.append(silhouette_avg)

        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Agglomerative Clustering')
        plt.xticks(cluster_range)
        st.pyplot()

        st.subheader("Threshold Analysis for Agglomerative Clustering")
        suggestions = threshold_clustering(silhouette_scores)
        for k, suggestion in enumerate(suggestions, start=2):
            st.write(suggestion)

        st.write("DBSCAN does not require specifying the number of clusters.")
        st.write("Hence, Elbow Method is not applicable.")

        st.subheader("Silhouette Score for DBSCAN Clustering")
        silhouette_scores = []

        eps_values = np.linspace(0.1, 2.0, num=10)
        min_samples_values = range(2, 10)
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(df)
                if len(np.unique(cluster_labels)) < 2:
                    silhouette_scores.append(np.nan)  # Skip silhouette score calculation
                else:
                    silhouette_avg = silhouette_score(df, cluster_labels)
                    silhouette_scores.append(silhouette_avg)

        silhouette_scores = np.array(silhouette_scores).reshape(len(eps_values), len(min_samples_values))

        plt.figure(figsize=(10, 6))
        for i, eps in enumerate(eps_values):
            plt.plot(min_samples_values, silhouette_scores[i], marker='o', label=f"Eps = {eps}")
        plt.xlabel('Min Samples')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for DBSCAN Clustering')
        plt.legend()
        st.pyplot()

####################### Neural Network ################################

@st.cache_data(ttl="2h")
def __init__(self,root_dir):
        self.root_dir = root_dir
        self.training_dir = os.path.join('directory','training')
        self.validation_dir = os.path.join('directory','validation')
        self.history = None
        self.train_gen = None
        self.val_gen = None

@st.cache_data(ttl="2h")
def make_train_val_dirs(self):
        os.makedirs('directory',exist_ok=True)
        os.makedirs(os.path.join('directory','training'),exist_ok=True)
        os.makedirs(os.path.join('directory','validation'),exist_ok=True)
        classes = os.listdir(self.root_dir)
        num_classes = len(classes)
        for class_name in classes:
            os.makedirs(os.path.join('directory','training',class_name),exist_ok=True)
            os.makedirs(os.path.join('directory','validation',class_name),exist_ok=True)
        print("Training and validation directories created!")

@st.cache_data(ttl="2h")
def create_dataset(self,split_size = 0.8):
        for dir_name in os.listdir(self.root_dir):
            samples = os.listdir(os.path.join(self.root_dir,dir_name))
            random.shuffle(samples)
            split_no = int(len(samples)*split_size)
            train_samples = samples[:split_no]
            val_samples = samples[split_no:]
            for sample in train_samples:
                s = os.path.join(self.root_dir,dir_name,sample)
                d = os.path.join('directory','training',dir_name,sample)
                shutil.copyfile(s,d)
                
            for sample in val_samples:
                s = os.path.join(self.root_dir,dir_name,sample)
                d = os.path.join('directory','validation',dir_name,sample)
                shutil.copyfile(s,d)

        print("Dataset created!")

@st.cache_data(ttl="2h")
def train_val_gens(self):
        mode = "categorical"
        
        train_datagen = ImageDataGenerator(rescale=1./255.,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
        train_generator = train_datagen.flow_from_directory(self.training_dir,
                                                            batch_size=20,
                                                            class_mode=mode,
                                                            target_size=(150,150))
        validation_datagen = ImageDataGenerator(rescale=1./255.)
                        
        validation_generator = validation_datagen.flow_from_directory(self.validation_dir,
                                                                    batch_size=20,
                                                                    class_mode=mode,
                                                                    target_size=(150,150))
        
        self.train_gen = train_generator
        self.val_gen = validation_generator

        print("Generators created !")
    
@st.cache_data(ttl="2h")
def model_vgg(self):
        num_classes = len(os.listdir(os.path.join('directory', 'training')))
        base_model = VGG16(weights = 'imagenet',include_top = False,input_shape = (150,150,3))
        for layer in base_model.layers:
            layer.trainable = False
            
        X = tensorflow.keras.layers.Flatten()(base_model.output)
        X = tensorflow.keras.layers.Dense(256,activation = 'relu')(X)
        output = tensorflow.keras.layers.Dense(num_classes,activation = 'softmax')(X)
        model = Model(inputs = base_model.input,outputs = output)
        model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy','precision','recall'])
        history = model.fit(self.train_gen,validation_data = self.val_gen,epochs = 10)
        self.history = history
        model.save('vgg_model.h5')
        print("Model is trained")

@st.cache_data(ttl="2h")
def model_mobilenetv2(self):
        num_classes = len(os.listdir(os.path.join('directory', 'training')))
        base_model = MobileNetV2(weights = 'imagenet',include_top = False,input_shape = (150,150,3))
        for layer in base_model.layers:
            layer.trainable = False
            
        X = base_model.output
        X = tensorflow.keras.layers.GlobalAveragePooling2D()(X)
        X = tensorflow.keras.layers.Dense(1024,activation='relu')(X)
        output = tensorflow.keras.layers.Dense(num_classes,activation = 'softmax')(X)
        model = Model(inputs = base_model.input,outputs = output)
        model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy','precision','recall'])
        history = model.fit(self.train_gen,validation_data = self.val_gen,epochs = 10)
        self.history = history
        model.save('mobilenet_model.h5')
        print("Model is trained")

@st.cache_data(ttl="2h")
def resnet(self):
        num_classes = len(os.listdir(os.path.join('directory', 'training')))
        base_model = ResNet50(weights = 'imagenet',include_top = False,input_shape = (150,150,3))
        for layer in base_model.layers:
            layer.trainable = False
            
        X = base_model.output
        X = tensorflow.keras.layers.GlobalAveragePooling2D()(X)
        X = tensorflow.keras.layers.Dense(1024,activation='relu')(X)
        output = tensorflow.keras.layers.Dense(num_classes,activation = 'softmax')(X)
        model = Model(inputs = base_model.input,outputs = output)
        model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy','precision','recall'])
        history = model.fit(self.train_gen,validation_data = self.val_gen,epochs = 10)
        self.history = history
        model.save('resnet_model.h5')
        print("Model is trained")        
        
#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns((0.15,0.85))
with col1:
        problem_type = st.selectbox("**Pick your Problem Type**", ["Regression", "Classification", "Clustering", "Image Classification"])
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
                img_zip_file = st.file_uploader("**Upload your Dataset**", type=['zip'])
        else:
                dataset_file = st.file_uploader("**Upload your Dataset**", type=['csv'])
    

        if state != 4:
            if dataset_file:
                df = pd.read_csv(dataset_file)
                if state == 1 or state == 2:
                    target_y = st.multiselect("**Target (Dependent) Variable**", df.columns)


