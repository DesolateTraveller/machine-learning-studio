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


