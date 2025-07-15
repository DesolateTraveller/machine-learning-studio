#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Template Graphics
#---------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
#----------------------------------------
# Plots
import altair as alt
import plotly.express as px
import plotly.offline as pyoff
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import scikitplot as skplt
#from scikitplot.metrics import plot_lift_curve, plot_cumulative_gain
from mpl_toolkits.mplot3d import Axes3D
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
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, f_regression, chi2, VarianceThreshold, mutual_info_regression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, silhouette_score
from sklearn.metrics import accuracy_score, auc, roc_auc_score, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef, precision_recall_curve
from yellowbrick.classifier import ROCAUC, ClassPredictionError
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
from sklearn.decomposition import PCA
#----------------------------------------
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from kmodes.kmodes import KModes
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, homogeneity_score, adjusted_rand_score, completeness_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
#---------------------------------------------------------------------------------------------------------------------------------
### Import Functions
#---------------------------------------------------------------------------------------------------------------------------------
from ml_studio_func import load_file, check_missing_values, check_outliers, handle_categorical_missing_values, handle_numerical_missing_values
from ml_studio_func import label_encode, onehot_encode, scale_features, calculate_vif, drop_high_vif_variables, iterative_vif_filtering
from ml_studio_func import check_linear_regression_assumptions, calculate_metrics, evaluate_model, evaluate_model_train
from ml_studio_func import wrap_labels
from ml_studio_func import plot_learning_curve, plot_validation_curve, plot_lift_curve, plot_gain_curve
from ml_studio_func import compute_psi, compute_drift_matrix
#---------------------------------------------------------------------------------------------------------------------------------
### Title for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
st.set_page_config(page_title="ML Studio | v1.0",
                   layout="wide",
                   page_icon="💻",              
                   initial_sidebar_state="auto")
#---------------------------------------------------------------------------------------------------------------------------------
### CSS
#---------------------------------------------------------------------------------------------------------------------------------
st.markdown(
        """
        <style>
        .centered-info {display: flex; justify-content: center; align-items: center; 
                        font-weight: bold; font-size: 15px; color: #007BFF; 
                        padding: 5px; background-color: #FFFFFF;  border-radius: 5px; border: 1px solid #007BFF;
                        margin-top: 0px;margin-bottom: 5px;}
        .stMarkdown {margin-top: 0px !important; padding-top: 0px !important;}                       
        </style>
        """,unsafe_allow_html=True,)

#---------------------------------------------------------------------------------------------------------------------------------
### Description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
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
    <div class="title-small">Version : 1.0</div>
    """,
    unsafe_allow_html=True
)

#----------------------------------------
st.markdown('<div class="centered-info"><span style="margin-left: 10px;">A lightweight Machine Learning (ML) streamlit app that help to analyse different types machine learning problems</span></div>',unsafe_allow_html=True,)
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
    #"LGBM Regressor": LGBMRegressor(),
    #"AdaBoost Regressor": AdaBoostRegressor(),
    "Extra Trees Regressor": ExtraTreesRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor()
}

#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------
ml_type = st.selectbox("**:blue[Select an anlgorithm]**", ["None", "Classification", "Clustering", "Regression",])                                                 

if ml_type == "None":
    st.warning("Please choose an algorithm to proceed with the analysis.")

#---------------------------------------------------------------------------------------------------------------------------------      
elif ml_type == "Classification":
    
    col1, col2= st.columns((0.15,0.85))
    with col1:           
        with st.container(border=True):
                
            clf_file = st.file_uploader("📁 **:blue[Choose file (for training)]**",type=["csv", "xls", "xlsx", "parquet"], accept_multiple_files=False, key="file_upload")
            if clf_file is not None:
                st.success("Data loaded successfully!")
                df = load_file(clf_file) 
                
                st.divider()
                
                df_new = None
                new_data_file = st.file_uploader("📁 **:blue[Choose file for prediction (optional)]**", type=["csv", "xlsx", "xls","parquet"], key="new_data") 
                if new_data_file:
                    df_new = load_file(new_data_file)
                    st.success("✅ New data uploaded successfully.")
                
                st.divider()

                target_variable = st.selectbox("**:blue[Choose Target Variable]**", options=["None"] + list(df.columns), key="target_variable")
                
                with st.expander("**✂️ Features for Deletion**", expanded=False):
                    optional_cols_to_delete = st.multiselect("**:blue[Feature Removal (optional)]**",
                    options=[col for col in df.columns if col != target_variable],
                    help="Choose columns to drop from dataset before proceeding.")
                if optional_cols_to_delete:
                    st.info(f"🗑️ Dropping {len(optional_cols_to_delete)} selected column(s): {optional_cols_to_delete}")
                    df = df.drop(columns=optional_cols_to_delete)
                else:
                    st.success("✅ No columns selected for deletion.")
                    
                if target_variable == "None":
                    st.warning("Please choose a target variable to proceed with the analysis.")                
                
                else:
                    #st.warning("Tune or Change the **Hyperparameters**(tab shown in the top) whenever required.")   
                    with col2:

                        with st.popover("**:blue[:hammer_and_wrench: Hyperparameters]**",disabled=False, use_container_width=True,help="Tune the hyperparameters whenever required"):

                            subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
                            with subcol1:      
                                    numerical_strategies = ['mean', 'median', 'most_frequent']
                                    categorical_strategies = ['constant','most_frequent']
                                    selected_numerical_strategy = st.selectbox("**Missing value treatment : Numerical**", numerical_strategies)
                                    selected_categorical_strategy = st.selectbox("**Missing value treatment : Categorical**", categorical_strategies) 
                                    st.divider() 
                                    treatment_option = st.selectbox("**Select a outlier treatment option**", ["Cap Outliers","Drop Outliers", ])

                            with subcol2: 
                                    scaling_reqd = st.selectbox("**Requirement of scalling**", ["no", "yes"])
                                    if scaling_reqd == 'yes':                       
                                        scaling_method = st.selectbox("**Scaling method**", ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
                                    if scaling_reqd == 'no':   
                                        scaling_method = 'N/A'
                                    st.divider()
                                    vif_threshold = st.number_input("**VIF Threshold**",2.0,5.0,5.0)                        
                                    st.divider()
                                    num_features_to_select = st.slider("**Number of Independent Features**",1,len(df.columns),5)
                                    st.divider()
                                    threshold = st.number_input("**Variance Threshold**",0.0,0.1,0.05)      

                            with subcol3:                    
                                    train = st.slider("**Train Size (as %)**", 10, 90, 70, 5)
                                    test = st.slider("**Test Size (as %)**", 10, 50, 30, 5)    
                                    random_state = st.number_input("**Random State**", 0, 100, 42)
                                    n_jobs = st.number_input("**Parallel Processing (n_jobs)**", -10, 10, 1) 
                                    st.divider()
                                    selected_metric_clf = st.selectbox("**Classification metrics | Best Model fitting**", ["Accuracy", "AUC", "Recall", "Precision", "F1 Score", "Kappa", "MCC"])

                            with subcol4: 
                                
                                with st.expander("**📌 Parameters | Logistic Regression**", expanded=False):
                                    penalty = st.selectbox("**Penalty**", ["l1", "l2", "elasticnet"]) 
                                    if penalty == "elasticnet":
                                       l1_ratio = st.slider("**Elastic-Net mixing parameter**", 0,1, 0.01, 0.05, 0.1)   
                                    solver= st.radio("**Solver**", ('liblinear', 'lbfgs'))
                                    C = st.slider("**C (Regularization)**", 0.01,10.0,0.01,1.0)
                                
                                with st.expander("**📌 Parameters | Tree & Boosting Method**", expanded=False):                                                                                                                                                
                                    n_estimators = st.slider("**Number of Estimators**", 10, 1000, 50, 100)
                                    criterion = st.selectbox("**Criteria**", ["gini", "entropy", "log_loss"]) 
                                    max_depth = st.slider("**Max Depth**", 1, 20, 1, 10)    
                                    loss = st.selectbox("**Loss Function**", ["log_loss","exponential"]) 
                                    #min_samples_split = st.slider("**Min Samples Split**", 2, 10, 1, 2)
                                    min_samples_leaf = st.slider("**Min Samples Leaf**", 2, 10, 1, 2)
                                    learning_rate = st.number_input("**Learning rate**",0.01,1.00,0.01,0.10, key ='learning_rate')
                                    #gamma =st.number_input("**min_split_loss**",0.00,1.00,0.05,0.1)
                                    #lamda = st.slider("**L2 reg**", 0, 100, 1, 10)
                                    alpha = st.slider("**L1 reg**", 0, 100, 1, 10)
                                    eval_metric = st.selectbox("**Evaluation Metrics**", ["logloss", "auc", "aucpr","ndcg","map"])

                                st.divider()
                                kernel = st.selectbox("**Kernel**", ["linear", "poly", "rbf", "sigmoid"])
                                gamma = st.selectbox("**Gamma**", ["scale", "auto"])
                                 
                            with subcol5: 
                                    k = st.slider("**Number of K folds**", min_value=3, max_value=10, step=1, value=5)  
                                    st.divider()
                                    search_method = st.selectbox("**🔍 Select Search Method**", ["Grid Search", "Randomized Search"])
                                    n_iter = st.slider("**Number of iterations for RandomizedSearchCV**", 1, 20, 10) if search_method == "Randomized Search" else None
                                    st.divider()
                                    scoring_clf = st.selectbox("**Select Scoring Metric**", ["accuracy", "f1", "roc_auc", "precision", "recall"])        

                        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        tabs = st.tabs(["**📊 Overview**","**📈 Visualizations**","**🔧 Preprocessing**","**✅ Assumptions**","**⚖️ Comparison**","**🎯 Importance**","**🔄 Validation**","**📈 Graph**","**📋 Results**","**🎲 Prediction**", "**⚠︎ Drift**"])
                        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        with tabs[0]:
                            
                            unwanted_substrings = ['unnamed', 'nan', 'deleted']
                            cols_to_delete = [col for col in df.columns if any(sub in col.lower() for sub in unwanted_substrings)]
                            if cols_to_delete:
                                st.warning(f"🗑️ {len(cols_to_delete)} column(s) deleted.")
                            else:
                                st.info("✅ No unwanted columns found. Nothing deleted after importing.")
                            df= df.drop(columns=cols_to_delete)
                            
                            st.info("✅ Showing Top 3 rows for reference.") 
                            st.dataframe(df.head(3))
                            
                            st.divider()
                            
                            with st.container(border=True):

                                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

                                col1.metric('**input values (rows)**', df.shape[0], help='number of rows')
                                col2.metric('**variables (columns)**', df.shape[1], help='number of columns')     
                                col3.metric('**numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
                                col4.metric('**categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')
                                col5.metric('**Missing values**', df.isnull().sum().sum(), help='Total missing values in the dataset')
                                col6.metric('**Target Variable**', target_variable, help='Selected target variable')  
                                unique_vals = df[target_variable].nunique()
                                if unique_vals == 2:
                                    target_type = "Binary"
                                else:
                                    target_type = "Multiclass"
                                col7.metric('**Type of Target Variable**', target_type, help='Classification problem type (binary/multiclass)')       
                                
                            with st.container(border=True):
                                
                                df_describe_table = df.describe(include='all').T.reset_index().rename(columns={'index': 'Feature'})
                                st.markdown("##### 📊 Descriptive Statistics")
                                st.dataframe(df_describe_table)

                        with tabs[1]:
                            
                            cat_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
                            num_vars = df.select_dtypes(include=['number']).columns.tolist()
                            
                            with st.container(border=True):
                                if cat_vars:
                                    st.success(f"📋 Categorical Variables Found: {len(cat_vars)}")
                                else:
                                    st.warning("⚠️ No categorical variables found.")

                                if cat_vars:
                                    for i in range(0, len(cat_vars), 3):
                                        cols = st.columns(3)
                                        for j, col_name in enumerate(cat_vars[i:i+3]):
                                            with cols[j]:
                                                fig, ax = plt.subplots(figsize=(5,2.5))
                                                df[col_name].value_counts().plot(kind='bar', ax=ax, color='skyblue')
                                                ax.set_title(f"{col_name}", fontsize=10)
                                                ax.set_ylabel("Count", fontsize=9)
                                                ax.set_xlabel("")
                                                ax.tick_params(axis='x', rotation=45, labelsize=8)
                                                ax.tick_params(axis='y', labelsize=8)
                                                st.pyplot(fig,use_container_width=True)

                            with st.container(border=True):
                                if num_vars:
                                    st.success(f" 📈 Numerical Variables Found: {len(num_vars)}")
                                else:
                                    st.warning("⚠️ No numerical variables found.")

                                if num_vars:
                                    for i in range(0, len(num_vars), 3):
                                        cols = st.columns(3)
                                        for j, col_name in enumerate(num_vars[i:i+3]):
                                            with cols[j]:
                                                st.markdown(f"**{col_name}**")
                                                skew_val = df[col_name].skew()
                                                skew_tag = (
                                                    "🟩 Symmetric" if abs(skew_val) < 0.5 else
                                                    "🟧 Moderate skew" if abs(skew_val) < 1 else
                                                    "🟥 Highly skewed"
                                                )
                                                st.info(f"Skewness: {skew_val:.2f} — {skew_tag}")

                                                fig_box, ax_box = plt.subplots(figsize=(4,2))
                                                sns.boxplot(y=df[col_name], ax=ax_box, color='lightcoral')
                                                ax_box.set_title("Box Plot",fontsize=8)
                                                ax_box.set_ylabel("", fontsize=8)
                                                ax_box.set_xlabel("", fontsize=8)
                                                ax_box.tick_params(axis='y', labelsize=8)
                                                st.pyplot(fig_box,use_container_width=True)

                                                fig_hist, ax_hist = plt.subplots(figsize=(4,2))
                                                sns.histplot(df[col_name], kde=True, ax=ax_hist, color='steelblue')
                                                ax_hist.set_title("Histogram", fontsize=8)
                                                ax_hist.set_xlabel("" ,fontsize=8)
                                                ax_hist.set_ylabel("", fontsize=8)
                                                ax_hist.tick_params(axis='x', labelsize=8)
                                                ax_hist.tick_params(axis='y', labelsize=8)
                                                st.pyplot(fig_hist,use_container_width=True)
                                                
                                        st.markdown('---')   

                            with st.container(border=True):
                                    
                                    numeric_df = df.select_dtypes(include=['number'])
                                    cmap = sns.diverging_palette(220,20,as_cmap=True)
                                    corrmat = numeric_df.corr()
                                    fig, ax = plt.subplots(figsize=(20,20))
                                    sns.heatmap(corrmat, annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax)
                                    ax.set_title("Correlation Heatmap", fontsize=14)
                                    plt.xticks(rotation=45, ha="right", fontsize=10)
                                    plt.yticks(rotation=0, fontsize=10)
                                    st.pyplot(fig, use_container_width=True)
                                    
                                    df_cor = df.copy()
                                    if target_variable in df_cor.columns:
                                        if target_variable in num_vars:
                                            num_vars.remove(target_variable)
                                        if not pd.api.types.is_numeric_dtype(df_cor[target_variable]):
                                            le = LabelEncoder()
                                            df_cor[target_variable] = le.fit_transform(df_cor[target_variable])
                                        numeric_cols = df_cor[num_vars + [target_variable]].select_dtypes(include=['number'])
                                        target_corr = numeric_cols.corr()[target_variable].drop(target_variable).reset_index()
                                        target_corr.columns = ["Variable", f"Correlation with {target_variable}"]
                                        target_corr = target_corr.sort_values(by=f"Correlation with {target_variable}", ascending=False)
                                        
                                        st.markdown(f"##### 🎯 Correlation of Variables with **{target_variable}**")
                                        st.dataframe(target_corr.reset_index(drop=True), hide_index=True)
                                    else:
                                        st.warning(f"⚠️ Target variable `{target_variable}` not found in the dataset.")
                                    
                        with tabs[2]:
                            
                            with st.container(border=True):
                                
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
                                            st.download_button("📥 Download Treated Data (.csv)", cleaned_df.to_csv(index=False), file_name="treated_data.csv")

                            with st.container(border=True):

                                if st.checkbox("Show Duplicate Values"):
                                        if missing_values.empty:
                                            st.table(df[df.duplicated()].head(2))
                                        else:
                                            st.table(cleaned_df[cleaned_df.duplicated()].head(2))

                            with st.container(border=True):

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
                                        st.dataframe(outliers,hide_index=True)
                    
                                with col2:
                                    
                                    if treatment_option == "Drop Outliers":
                                        df = df[~outliers['Column'].isin(outliers[outliers['Number of Outliers'] > 0]['Column'])]
                                        st.success("Outliers dropped. Preview of the cleaned dataset:")
                                        st.table(df.head())
                                    
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
                                            st.dataframe(df,hide_index=True)     

                            with st.container(border=True):

                                    categorical_columns = df.select_dtypes(include=['object']).columns
                                    if len(categorical_columns) == 0:
                                        st.warning("There are no categorical variables in the dataset.Proceed with the original DataFrame")
                                        df = df.copy()
                                    else:
                                        for feature in df.columns: 
                                            if df[feature].dtype == 'object': 
                                                print('\n')
                                                print('feature:',feature)
                                                print(pd.Categorical(df[feature].unique()))
                                                print(pd.Categorical(df[feature].unique()).codes)
                                                df[feature] = pd.Categorical(df[feature]).codes
                                        st.success("**Categorical variables are encoded**")
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(label="📥 Download Encoded Data (for review) (.csv)", data=csv, file_name='encoded_data.csv', mime='text/csv')

                            with st.container(border=True):
                                
                                    if scaling_reqd == 'yes':     
                                        df = scale_features(df,scaling_method)
                                        st.success("**Data is scaled for further treatment**")
                                        csv = df.to_csv(index=False).encode('utf-8')
                                        st.download_button(label="📥 Download Scaled Data (for review) (.csv)", data=csv, file_name='scaled_data.csv', mime='text/csv')
                                    else:
                                        st.warning("Data is not scaled, orginal data is considered for further treatment | If Scalling required, change the options in the drop-down menu from **Hyperparameter** tab in the top.")
                                    #st.dataframe(df.head()) 

                            with st.container(border=True):                                        

                                X = df.drop(columns=target_variable)
                                y = df[target_variable]
                                st.markdown(f"""
                                **Original number of features**: **{X.shape[1]}**

                                🧪 The following feature selection methods are applied sequentially to identify the most optimal features:
                                - Variance Threshold
                                - SelectKBest 
                                - VIF (Variance Influence Factor)
                                """)
                                st.divider()
                                col1, col2, col3 = st.columns(3)
                                with col1:

                                        st.markdown("**Feature Selection Method : VarianceThreshold**")

                                        X = df.drop(columns=target_variable)
                                        selector = VarianceThreshold(threshold=threshold)
                                        X_varth = selector.fit_transform(X)
                                        selected_varth_indices = selector.get_support(indices=True)
                                        selected_varth_features = X.columns[selected_varth_indices]
                                        feature_variances = selector.variances_
                                        variance_df = pd.DataFrame({
                                            "Feature": X.columns,
                                            "Variance": feature_variances,
                                            "Status": ["✅ Kept" if i in selected_varth_indices else "❌ Dropped" for i in range(len(X.columns))]
                                        }).sort_values(by="Variance", ascending=True)
                                        st.write(f"✅ Number of features before Variance Threshold: **{X.shape[1]}**")
                                        st.dataframe(variance_df, hide_index=True)

                                        # Remarks
                                        dropped_features = variance_df[variance_df["Status"] == "❌ Dropped"]
                                        st.markdown(f"""
                                        ℹ️ **Remarks:**  
                                        - Threshold for variance filtering: **{threshold}**  
                                        - Total features before filtering: **{X.shape[1]}**  
                                        - Features dropped due to low variance: **{len(dropped_features)}**

                                        These features had variance below the set threshold and were considered non-informative.
                                        """)

                                with col2:
                  
                                        st.markdown("**Feature Selection Method : Selectkbest**")          

                                        st.markdown(f"✅Number of features before SelectKBest method: **{len(selected_varth_features)}**")
                                        X_kbest_input = df[selected_varth_features].copy()
                                        y = df[target_variable]
                                        num_features_to_select = X_kbest_input.shape[1]
                                        f_selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
                                        X_f_selected = f_selector.fit_transform(X_kbest_input, y)
                                        f_scores = f_selector.scores_
                                        f_pvalues = f_selector.pvalues_
                                        mi_scores = mutual_info_regression(X_kbest_input, y, random_state=42)
                                        selection_df = pd.DataFrame({
                                            "Feature": selected_varth_features,
                                            "F-Score": f_scores,
                                            "p-value": f_pvalues,
                                            "Mutual Info Score": mi_scores
                                        }).sort_values(by="F-Score", ascending=False)
                                        st.dataframe(selection_df, hide_index=True)
                                        pval_threshold = 0.05
                                        filtered_selection_df = selection_df[selection_df["p-value"] < pval_threshold].copy()
                                        st.markdown(f"""
                                        ℹ️ **Remarks:**  
                                        - Total features before p-value filtering: **{len(selection_df)}**  
                                        - Features with **p-value < {pval_threshold}** retained: **{len(filtered_selection_df)}**  
                                        - These are passed to the next step (**VIF filtering**).
                                        """)
                                        st.divider()
                                        st.markdown("**Filtered Features (Significant by F-test):**")
                                        st.dataframe(filtered_selection_df, hide_index=True)

                                with col3:
                                                                            
                                        st.markdown("**Feature Selection Method : VIF**")  

                                        selected_kbest_features = filtered_selection_df["Feature"].tolist()
                                        X_vif_input = df[selected_kbest_features].copy()
                                        filtered_vif_df, final_vif_table = iterative_vif_filtering(X_vif_input, threshold=vif_threshold)
                                        st.write(f"✅ Final number of features after VIF filtering: {filtered_vif_df.shape[1]}")
                                        st.dataframe(final_vif_table, hide_index=True) 

                        with tabs[3]: 
                            
                            selected_features = filtered_vif_df.columns.tolist()
                            X = filtered_vif_df.copy()
                            y = df[target_variable]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

                            with st.container(border=True):
                                st.markdown(f"""
                                ✅ **Final dataset prepared for modeling**  
                                - Number of selected features: **{len(selected_features)}**  
                                - Train set size: **{X_train.shape[0]} rows**  
                                - Test set size: **{X_test.shape[0]} rows**
                                """) 
                            
                            st.success("**No assumptions for classification problems**")
                                                                        
                        with tabs[4]: 
                            
                            models = {
                                        "Logistic Regression": LogisticRegression(penalty=penalty, C=C, solver=solver),
                                        "Ridge Classifier": RidgeClassifier(),
                                        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                                        "Decision Tree Classifier": DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf),
                                        "Random Forest Classifier": RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth),
                                        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate,loss=loss,min_samples_leaf=min_samples_leaf),
                                        "Ada Boost Classifier": AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate,),
                                        "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=n_estimators,criterion=criterion, max_depth=max_depth,min_samples_leaf=min_samples_leaf),
                                        "Light Gradient Boosting Machine": LGBMClassifier(),
                                        #"K Neighbors Classifier": KNeighborsClassifier(),
                                        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
                                        "Extreme Gradient Boosting": XGBClassifier(eta=learning_rate,max_depth=max_depth,alpha=alpha,eval_metric=eval_metric),  
                                                                                
                                        #"Naive Bayes": GaussianNB(),
                                        #"CatBoost Classifier": CatBoostClassifier(verbose=0),
                                        #"Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                                        #"SVM - Linear Kernel": SVC(kernel="linear", probability=True)
                                        }  
                                            
                            with st.spinner("Setting up and comparing models..."):

                                if target_type == "Binary":

                                    results = []
                                    for name, model in models.items():
                                                metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
                                                metrics["Model"] = name
                                                results.append(metrics)
                                    results_df = pd.DataFrame(results)

                                    results_train = []
                                    for name, model in models.items():
                                                metrics = evaluate_model_train(model, X_train, X_test, y_train, y_test)
                                                metrics["Model"] = name
                                                results_train.append(metrics)
                                    results_df_train = pd.DataFrame(results_train)
                                                                                
                                    with st.container(border=True):                                    
                                    
                                        col1, col2= st.columns(2)
                                        with col1: 
                                                st.markdown("##### Model Metrices Comparison | Train Dataset")
                                                st.dataframe(results_df,hide_index=True)
                                            
                                        with col2:
                                            st.markdown("##### Model Metrices Comparison | Test Dataset") 
                                            st.dataframe(results_df_train ,hide_index=True)    
                                                                            
                                    with st.expander("**🏆 Best Model per Metric**", expanded=False): 
                                         
                                        col1, col2= st.columns(2)
                                        with col1:
                                                st.markdown("##### Train Dataset")                                  
                                                metric_cols = [col for col in results_df.columns if col != "Model"]
                                                best_models_per_metric = {
                                                    metric: results_df.loc[results_df[metric].idxmax(), "Model"]
                                                    for metric in metric_cols}
                                                summary_df = pd.DataFrame({"Metric": list(best_models_per_metric.keys()),"Best Model": list(best_models_per_metric.values())})
                                                st.dataframe(summary_df, hide_index=True) 

                                        with col2:
                                                st.markdown("##### Test Dataset")                                  
                                                metric_cols = [col for col in results_df_train.columns if col != "Model"]
                                                best_models_per_metric = {
                                                    metric: results_df_train.loc[results_df_train[metric].idxmax(), "Model"]
                                                    for metric in metric_cols}
                                                summary_df_train = pd.DataFrame({"Metric": list(best_models_per_metric.keys()),"Best Model": list(best_models_per_metric.values())})
                                                st.dataframe(summary_df_train, hide_index=True) 
                                                
                                    best_model_clf = best_models_per_metric[selected_metric_clf]
                                    best_model = models[best_model_clf]
                                    best_model.fit(X_train, y_train)
                                    
                                    y_pred_train_best = best_model.predict(X_train)
                                    y_proba_train_best = best_model.predict_proba(X_train)[:, 1] if hasattr(best_model, "predict_proba") else None                                    
                                    y_pred_best = best_model.predict(X_test)
                                    y_proba_best = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
                                                                                    
                                    with st.container(border=True):

                                        metrics = ["Accuracy", "AUC", "Recall", "Precision", "F1 Score", "Kappa", "MCC"]
                                        metrics_df_train = results_df_train.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
                                        metrics_df_test = results_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")

                                        for metric in metrics:
                                            st.markdown(f"##### 📊 **{metric}** — Train vs Test Comparison")
                                            col1, col2 = st.columns(2)

                                            train_vals = metrics_df_train[metrics_df_train["Metric"] == metric].set_index("Model")["Value"]
                                            test_vals = metrics_df_test[metrics_df_test["Metric"] == metric].set_index("Model")["Value"]
                                            delta_vals = (test_vals - train_vals).round(2)

                                            with col1:
                                                st.markdown(f"##### 🟦 Train — **{metric}**")
                                                fig, ax = plt.subplots(figsize=(6,2.5))
                                                bars = sns.barplot(x="Model", y="Value",data=metrics_df_train[metrics_df_train["Metric"] == metric],color="steelblue", ax=ax)
                                                for bar in bars.patches:
                                                    height = bar.get_height()
                                                    ax.annotate(f"{height:.2f}",(bar.get_x() + bar.get_width() / 2, height),ha='center', va='bottom', fontsize=8, color='black')
                                                ax.set_xlabel("", fontsize=8)
                                                ax.set_ylabel(metric, fontsize=8)
                                                ax.set_title(f"Train: {metric}", fontsize=8)
                                                ax.tick_params(axis='x', rotation=90, labelsize=8)
                                                st.pyplot(fig, use_container_width=True)

                                            with col2:
                                                st.markdown(f"##### 🟨 Test + Δ — **{metric}**")
                                                fig, ax = plt.subplots(figsize=(6,2.5))
                                                bars = sns.barplot(x="Model", y="Value",data=metrics_df_test[metrics_df_test["Metric"] == metric],color="darkorange", ax=ax)

                                                model_names = metrics_df_test[metrics_df_test["Metric"] == metric]["Model"].tolist()
                                                for i, bar in enumerate(bars.patches):
                                                    height = bar.get_height()
                                                    model = model_names[i]
                                                    delta = delta_vals.get(model, 0.0)
                                                    delta_str = f"Δ +{delta:.2f}" if delta > 0 else f"Δ {delta:.2f}"
                                                    color = 'green' if delta > 0 else 'red'

                                                    ax.annotate(f"{height:.2f}",(bar.get_x() + bar.get_width() / 2, height),ha='center', va='bottom', fontsize=8, color='black')
                                                    ax.annotate(delta_str,(bar.get_x() + bar.get_width() / 2, height + 0.02 * height),ha='center', va='top', fontsize=8, color=color)

                                                ax.set_xlabel("", fontsize=8)
                                                ax.set_ylabel(metric, fontsize=8)
                                                ax.set_title(f"Test: {metric}", fontsize=8)
                                                ax.tick_params(axis='x', rotation=90, labelsize=8)
                                                st.pyplot(fig, use_container_width=True)

                                elif target_type == "MultiClass":
                            
                                    results = []
                                    for name, model in models.items():
                                                metrics = evaluate_model(model, X_train, X_test, y_train, y_test, multi_class=True)
                                                metrics["Model"] = name
                                                results.append(metrics)
                                    results_df = pd.DataFrame(results)
                                    
                                    results_train = []
                                    for name, model in models.items():
                                                metrics = evaluate_model_train(model, X_train, X_test, y_train, y_test, multi_class=True)
                                                metrics["Model"] = name
                                                results_train.append(metrics)
                                    results_df_train = pd.DataFrame(results_train)   
                                                                             
                                    with st.container(border=True):                                    
                                    
                                        col1, col2= st.columns(2)
                                        with col1: 
                                                st.dataframe(results_df,hide_index=True)
                                            
                                        with col2:
                                            st.dataframe(results_df_train ,hide_index=True)    
                                                                            
                                    with st.expander("**🏆 Best Model per Metric**", expanded=False): 
                                         
                                        col1, col2= st.columns(2)
                                        with col1:
                                                st.markdown("##### Model Metrices | Train Dataset")                                  
                                                metric_cols = [col for col in results_df.columns if col != "Model"]
                                                best_models_per_metric = {
                                                    metric: results_df.loc[results_df[metric].idxmax(), "Model"]
                                                    for metric in metric_cols}
                                                summary_df = pd.DataFrame({"Metric": list(best_models_per_metric.keys()),"Best Model": list(best_models_per_metric.values())})
                                                st.dataframe(summary_df, hide_index=True) 

                                        with col2:
                                                st.markdown("##### Model Metrices | Test Dataset")                                  
                                                metric_cols = [col for col in results_df_train.columns if col != "Model"]
                                                best_models_per_metric = {
                                                    metric: results_df_train.loc[results_df_train[metric].idxmax(), "Model"]
                                                    for metric in metric_cols}
                                                summary_df_train = pd.DataFrame({"Metric": list(best_models_per_metric.keys()),"Best Model": list(best_models_per_metric.values())})
                                                st.dataframe(summary_df_train, hide_index=True) 
                                                
                                    best_model_clf = best_models_per_metric[selected_metric_clf]
                                    best_model = models[best_model_clf]
                                    best_model.fit(X_train, y_train)
                                    
                                    y_pred_train_best = best_model.predict(X_train)
                                    y_proba_train_best = best_model.predict_proba(X_train)[:, 1] if hasattr(best_model, "predict_proba") else None                                      
                                    y_pred_best = best_model.predict(X_test)
                                    y_proba_best = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
                                                                                    
                                    with st.container(border=True):

                                        metrics = ["Accuracy", "AUC", "Recall", "Precision", "F1 Score", "Kappa", "MCC"]
                                        metrics_df_train = results_df_train.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
                                        metrics_df_test = results_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")

                                        for metric in metrics:
                                            col1, col2 = st.columns(2)

                                            with col1:
                                                st.markdown("##### 🟦 Train")
                                                fig, ax = plt.subplots(figsize=(6,2.5))
                                                bars = sns.barplot(x="Model", y="Value",data=metrics_df_train[metrics_df_train["Metric"] == metric],color="steelblue", ax=ax)
                                                for bar in bars.patches:
                                                    height = bar.get_height()
                                                    bars.annotate(f"{height:.2f}",
                                                                (bar.get_x() + bar.get_width() / 2, height),
                                                                ha='center', va='bottom', fontsize=8, color='black')
                                                ax.set_xlabel("")
                                                ax.set_ylabel(metric, fontsize=9)
                                                ax.set_title(f"{metric}", fontsize=9)
                                                ax.tick_params(axis='x', rotation=90, labelsize=8)
                                                st.pyplot(fig, use_container_width=True)

                                            with col2:
                                                st.markdown("##### 🟨 Test")
                                                fig, ax = plt.subplots(figsize=(6,2.5))
                                                bars = sns.barplot(x="Model", y="Value",data=metrics_df_test[metrics_df_test["Metric"] == metric],color="darkorange", ax=ax)
                                                for bar in bars.patches:
                                                    height = bar.get_height()
                                                    bars.annotate(f"{height:.2f}",
                                                                (bar.get_x() + bar.get_width() / 2, height),
                                                                ha='center', va='bottom', fontsize=8, color='black')
                                                ax.set_xlabel("")
                                                ax.set_ylabel(metric, fontsize=9)
                                                ax.set_title(f"{metric}", fontsize=9)
                                                ax.tick_params(axis='x', rotation=90, labelsize=8)
                                                st.pyplot(fig, use_container_width=True)

                        with tabs[5]:
                            
                                    if best_model_clf == "Logistic Regression":
                                        importance_df = pd.DataFrame({'Feature': X.columns,'Importance': best_model.coef_.flatten()})
                                        importance_df['Percentage'] = (abs(importance_df['Importance']) / abs(importance_df['Importance']).sum()) * 100
                                        importance_df = importance_df.sort_values(by='Importance', ascending=False)

                                    elif hasattr(best_model, "feature_importances_"):
                                        importance_df = pd.DataFrame({"Feature": selected_features,"Importance": best_model.feature_importances_})
                                        importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
                                        importance_df = importance_df.sort_values(by='Importance', ascending=False)

                                    else:
                                        st.warning(f"⚠️ The selected model (**{best_model_clf}**) does not support native feature importance.")
                                        importance_df = pd.DataFrame({"Feature": selected_features})
                                        importance_df["Importance"] = None
                                        importance_df["Percentage"] = None

                                    col1, col2 = st.columns((0.25, 0.75))

                                    with col1:
                                        with st.container(border=True):
                                            st.markdown("#### 📋 Feature Importance Table")
                                            st.dataframe(importance_df, hide_index=True, use_container_width=True)

                                    with col2:
                                        with st.container(border=True):
                                            st.markdown("#### 📊 Feature Importance Plot")
                                            if "Importance" in importance_df.columns and importance_df["Importance"].notnull().all():
                                                plot_data_imp = [go.Bar(x=importance_df["Feature"],y=importance_df["Importance"],marker=dict(color='steelblue'))]
                                                plot_layout_imp = go.Layout(xaxis={"title": "Feature", "tickangle": -45},yaxis={"title": "Importance"},title="Feature Importance",margin=dict(t=30, b=80))
                                                fig = go.Figure(data=plot_data_imp, layout=plot_layout_imp)
                                                st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                st.info("ℹ️ Plot not available for this model.")

                                    top_n = min(5, len(importance_df.dropna()))
                                    top_features_text = "\n".join([
                                        f"{i+1}. {importance_df.iloc[i]['Feature']} ({importance_df.iloc[i]['Percentage']:.2f}%)"
                                        for i in range(top_n)
                                    ]) if "Percentage" in importance_df.columns else "N/A"

                                    with st.expander("📌 Top Features Summary", expanded=True):
                                        if top_n > 0 and "Percentage" in importance_df.columns:
                                            st.info(f"**Top {top_n} Features based on Importance:**\n{top_features_text}")
                                        else:
                                            st.warning("⚠️ No valid feature importance available to summarize.")

                                    with st.container(border=True):
                                        
                                            st.markdown("##### 🌐 SHAP Explanation (Top Features)")
                                            #explainer = shap.Explainer(best_model, X)
                                            #shap_values = explainer(X)
                                            #fig, ax = plt.subplots(figsize=(20,5))
                                            #shap.plots.bar(shap_values, max_display=10, show=False)
                                            #plt.tight_layout()
                                            #st.pyplot(fig)

                        with tabs[6]:   
                                                                      
                            col1, col2 = st.columns(2)

                            with col1:
                                with st.container(border=True):
                                    kf = KFold(n_splits=k, shuffle=True, random_state=42)
                                    scores = cross_val_score(best_model, X_train, y_train, cv=kf, scoring=scoring_clf)
                                    mean_score = np.mean(scores)
                                    std_score = np.std(scores)
                                    results_val = pd.DataFrame({"Fold": list(range(1, k + 1)),"Score": [round(s, 4) for s in scores]})
                                    #st.markdown("##### 🔁 Fold-wise Scores")
                                    st.dataframe(results_val, hide_index=True)
                                    metric_name = scoring_clf.upper()
                                    
                            with col2:
                                with st.container(border=True):
                                    st.markdown("##### 📊 Remarks")

                                    performance_stability = "stable" if std_score < 0.02 else "moderate" if std_score < 0.05 else "variable"

                                    explanation = f"""
                                    - Metric used: **{metric_name}**
                                    - In classification, **higher is better** (max = 1.0)
                                    - Average score across folds: **{mean_score:.4f}**
                                    - Standard deviation: **{std_score:.4f}** → **{performance_stability} performance**
                                    - Results may vary depending on class balance and folds used.

                                    > 📌 Use **Precision, Recall, F1** alongside Accuracy for imbalanced datasets.
                                    """
                                    st.markdown(explanation)      

                            with st.container(border=True):
                                    #st.markdown("##### 📊 Performance Summary")
                                    st.info(
                                        f"**{best_model_clf}** achieved an average **{mean_score:.4f}** "
                                        f"(± **{std_score:.4f}**) using **{k}-Fold Cross-Validation**.")

                            with st.container(border=True):

                                #base_model = models[best_model_clf]
                                st.divider()
                                #param_grid = {
                                    #"n_estimators": [100, 200, 300, 400],
                                    #"max_depth": [None, 5, 10, 20, 30],
                                    #"min_samples_split": [2, 5, 10],
                                    #"min_samples_leaf": [1, 2, 4],
                                    #"max_features": ["sqrt", "log2", None],
                                    #"bootstrap": [True, False]
                                #}

                                #param_dist = {
                                    #"n_estimators": [int(x) for x in range(50, 500, 50)],
                                    #"max_depth": [None] + list(range(5, 31, 5)),
                                    #"min_samples_split": [2, 3, 5, 7, 10],
                                    #"min_samples_leaf": [1, 2, 3, 4],
                                    #"max_features": ["sqrt", "log2", None],
                                    #"bootstrap": [True, False]
                                #}

                                #with st.spinner("Optimizing... please wait..."):
                                        #if search_method == "Grid Search":
                                            #search = GridSearchCV(estimator=base_model,param_grid=param_grid,scoring=scoring_clf,cv=k,n_jobs=-1,verbose=0)
                                        #else:
                                            #search = RandomizedSearchCV(estimator=base_model,param_distributions=param_dist,n_iter=n_iter,scoring=scoring_clf,cv=k,random_state=42,n_jobs=-1,verbose=0)

                                        #search.fit(X_train, y_train)
                                        #best_model_tuned = search.best_estimator_
                                        #best_params = search.best_params_
                                        #best_score = search.best_score_

                                        #st.markdown("##### 🧪 Best Parameters Found")
                                        #best_params_df = pd.DataFrame(best_params.items(), columns=["Hyperparameter", "Best Value"])
                                        #st.dataframe(best_params_df, hide_index=True)

                        with tabs[7]:

                            if target_type == 'Binary':

                                with st.container(border=True):
                                    
                                        col1, col2 = st.columns(2)  
                                        with col1:
                                                st.markdown("##### Classification Report | Train Dataset")  
                                                report = classification_report(y_train, y_pred_train_best, output_dict=True)
                                                report_df_tr = pd.DataFrame(report).transpose()
                                                st.table(report_df_tr)
                                                
                                        
                                        with col2: 
                                                st.markdown("##### Classification Report | Test Dataset")  
                                                report = classification_report(y_test, y_pred_best, output_dict=True)
                                                report_df = pd.DataFrame(report).transpose()
                                                st.table(report_df)
                                                
                                with st.container(border=True):
                                    
                                        col1, col2 = st.columns(2)  
                                        with col1: 
                                                cm = confusion_matrix(y_train, y_pred_train_best)
                                                plt.figure(figsize=(8,3))
                                                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                                                plt.title(f"Confusion Matrix | Train Dataset", fontsize=8)
                                                plt.xlabel("Predicted")
                                                plt.ylabel("Actual")
                                                st.pyplot(plt,use_container_width=True)
                                                                                                                                                               
                                        with col2:
                                                cm = confusion_matrix(y_test, y_pred_best)
                                                plt.figure(figsize=(8,3))
                                                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                                                plt.title(f"Confusion Matrix | Test Dataset", fontsize=8)
                                                plt.xlabel("Predicted")
                                                plt.ylabel("Actual")
                                                st.pyplot(plt,use_container_width=True)

                                with st.container(border=True):      
                        
                                        col1, col2 = st.columns(2)  
                                        with col1: 
                                                fpr, tpr, _ = roc_curve(y_train, y_proba_train_best)
                                                plt.figure(figsize=(8,3))
                                                plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {auc(fpr, tpr):.2f}")
                                                plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
                                                plt.xlabel("False Positive Rate")
                                                plt.ylabel("True Positive Rate")
                                                plt.title(f"AUC Curve | Train dataset", fontsize=8)
                                                plt.legend(loc="lower right")
                                                st.pyplot(plt,use_container_width=True)                                            
                                            
                                        with col2:                                     
                                                fpr, tpr, _ = roc_curve(y_test, y_proba_best)
                                                plt.figure(figsize=(8,3))
                                                plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {auc(fpr, tpr):.2f}")
                                                plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
                                                plt.xlabel("False Positive Rate")
                                                plt.ylabel("True Positive Rate")
                                                plt.title(f"AUC Curve | Test dataset", fontsize=8)
                                                plt.legend(loc="lower right")
                                                st.pyplot(plt,use_container_width=True)
                                                
                                with st.container(border=True):      
                        
                                        col1, col2 = st.columns(2)  
                                        with col1: 
                                                precisions, recalls, _ = precision_recall_curve(y_train, y_proba_train_best)
                                                plt.figure(figsize=(8,3))
                                                plt.plot(recalls, precisions, color="purple", lw=2)
                                                plt.xlabel("Recall")
                                                plt.ylabel("Precision")
                                                plt.title(f"Precision-Recall Curve | Train dataset", fontsize=8)
                                                st.pyplot(plt,use_container_width=True)                                            
                                            
                                        with col2:                                            
                                                precisions, recalls, _ = precision_recall_curve(y_test, y_proba_best)
                                                plt.figure(figsize=(8,3))
                                                plt.plot(recalls, precisions, color="purple", lw=2)
                                                plt.xlabel("Recall")
                                                plt.ylabel("Precision")
                                                plt.title(f"Precision-Recall Curve | Test Dataset", fontsize=8)
                                                st.pyplot(plt,use_container_width=True)

                                with st.container(border=True):      
                        
                                        col1, col2 = st.columns(2)  
                                        with col1: 
                                                precisions, recalls, thresholds = precision_recall_curve(y_train, y_proba_train_best)
                                                plt.figure(figsize=(8,3))
                                                plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
                                                plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
                                                plt.xlabel("Threshold")
                                                plt.title(f"Discrimination Threshold | Train dataset", fontsize=8)
                                                plt.legend(loc="best")
                                                st.pyplot(plt,use_container_width=True)                                            
                                            
                                        with col2:                                             
                                                precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_best)
                                                plt.figure(figsize=(8,3))
                                                plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
                                                plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
                                                plt.xlabel("Threshold")
                                                plt.title(f"Discrimination Threshold | Test Dataset", fontsize=8)
                                                plt.legend(loc="best")
                                                st.pyplot(plt,use_container_width=True)

                                with st.container(border=True):      
                        
                                        col1, col2 = st.columns(2)  
                                        with col1:
                                                fig1, ax1 = plt.subplots(figsize=(8,3))
                                                plot_lift_curve(y_train, y_proba_train_best, ax1)
                                                plt.title(f"Lift Curve | Train Dataset", fontsize=8)
                                                st.pyplot(fig1, use_container_width=True)

                                        with col2:
                                                fig1, ax1 = plt.subplots(figsize=(8,3))
                                                plot_lift_curve(y_test, y_proba_best, ax1)
                                                plt.title(f"Lift Curve | Test Dataset", fontsize=8)
                                                st.pyplot(fig1, use_container_width=True)

                                with st.container(border=True):      
                        
                                        col1, col2 = st.columns(2)  
                                        with col1:    
                                                fig2, ax2 = plt.subplots(figsize=(8,3))
                                                plot_gain_curve(y_train, y_proba_train_best, ax2)
                                                plt.title(f"Gain Curve | Train Dataset", fontsize=8)
                                                st.pyplot(fig2, use_container_width=True)                                            
                                            
                                        with col2:                                                                                       
                                                fig2, ax2 = plt.subplots(figsize=(8,3))
                                                plot_gain_curve(y_test, y_proba_best, ax2)
                                                plt.title(f"Gain Curve | Test Dataset", fontsize=8)
                                                st.pyplot(fig2, use_container_width=True)

                            if target_type == 'MultiClass':

                                with st.container(border=True):
                                    
                                        col1, col2 = st.columns(2)  
                                        with col1:
                                                st.markdown("##### Classification Report | Train Dataset")  
                                                report = classification_report(y_train, y_pred_train_best, output_dict=True)
                                                report_df_tr = pd.DataFrame(report).transpose()
                                                st.table(report_df_tr)
                                                
                                        
                                        with col2: 
                                                st.markdown("##### Classification Report | Test Dataset")  
                                                report = classification_report(y_test, y_pred_best, output_dict=True)
                                                report_df = pd.DataFrame(report).transpose()
                                                st.table(report_df)
                                                
                                with st.container(border=True):
                                    
                                        col1, col2 = st.columns(2)  
                                        with col1: 
                                                cm = confusion_matrix(y_train, y_pred_train_best)
                                                plt.figure(figsize=(8,3))
                                                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                                                plt.title(f"Confusion Matrix | Train Dataset", fontsize=8)
                                                plt.xlabel("Predicted")
                                                plt.ylabel("Actual")
                                                st.pyplot(plt,use_container_width=True)
                                                                                                                                                               
                                        with col2:
                                                cm = confusion_matrix(y_test, y_pred_best)
                                                plt.figure(figsize=(8,3))
                                                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                                                plt.title(f"Confusion Matrix | Test Dataset", fontsize=8)
                                                plt.xlabel("Predicted")
                                                plt.ylabel("Actual")
                                                st.pyplot(plt,use_container_width=True)

                        with tabs[8]:     
                            
                                    best_metrics=results_df.loc[results_df["Model"] == best_model_clf].iloc[0].to_dict()
                                    final_results_df = pd.DataFrame({"Metric": ["Type of Problem",
                                                                    "Target Variable",
                                                                    "Type of Target",
                                                                    "Scaling Method", 
                                                                    #"Feature Selection",
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
                                                                #f_sel_method,
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
                                        with st.container(border=True):

                                            st.dataframe(final_results_df, hide_index=True, use_container_width=True)
                                            #st.table(final_results_df)

                                    with col2:
                                        with st.container(border=True):
                                                                           
                                            X_test_results = X_test.copy()  
                                            X_test_results["Actual"] = y_test
                                            X_test_results["Predicted Label"] = y_pred_best
                                            if y_proba_best is not None:
                                                if target_type == "Binary":
                                                    X_test_results["Prediction Score"] = y_proba_best  # For binary classification, use the second column of predict_proba
                                                else:
                                                    for i in range(y_proba_best.shape[1]):
                                                        X_test_results[f"Class {i} Probability"] = y_proba_best[:, i]
                                            st.dataframe(X_test_results, use_container_width=True)
                                            st.download_button(label="📥 Download predicted data (.csv)",data=X_test_results.to_csv(index=False),file_name="classification_predictions.csv",mime="text/csv")     
 
                        with tabs[9]:   
                            
                                    if new_data_file and 'importance_df' in locals() and 'best_model' in locals():
                                        
                                        try:
                                            
                                            st.markdown("##### New Dataset (for prediction)")
                                            unwanted_substrings = ['unnamed', '0', 'nan', 'deleted']
                                            cols_to_delete = [col for col in df_new.columns if any(sub in col.lower() for sub in unwanted_substrings)]
                                            if cols_to_delete:
                                                st.warning(f"🗑️ {len(cols_to_delete)} column(s) deleted.")
                                            else:
                                                st.info("✅ No unwanted columns found. Nothing deleted after importing.")
                                                
                                            df_new= df_new.drop(columns=cols_to_delete)
                                            st.table(df_new.head(2))
                                            
                                            with st.status("**:blue[Collecting Information & Analyzing...]**", expanded=False) as status:
                                                
                                                st.write("Feature Cleaning...")
                                                
                                                missing_values = check_missing_values(df_new)
                                                cleaned_df = handle_numerical_missing_values(df_new, numerical_strategy='mean')
                                                cleaned_df = handle_categorical_missing_values(cleaned_df, categorical_strategy='most_frequent')   
                                                if missing_values.empty:
                                                    df_new = df_new.copy()
                                                else:
                                                    df_new = cleaned_df.copy()
                                                    
                                                outliers = check_outliers(df_new)
                                                df_new = df_new.copy()
                                                for column in outliers['Column'].unique():
                                                    Q1 = df_new[column].quantile(0.25)
                                                    Q3 = df_new[column].quantile(0.75)
                                                    IQR = Q3 - Q1
                                                    threshold = 1.5
                                                    df_new[column] = np.where(df_new[column] < Q1 - threshold * IQR, Q1 - threshold * IQR, df_new[column])
                                                    df_new[column] = np.where(df_new[column] > Q3 + threshold * IQR, Q3 + threshold * IQR, df_new[column])

                                                time.sleep(2)
                                                st.write("Feature Encoding...")
                                                
                                                categorical_columns = df_new.select_dtypes(include=['object']).columns
                                                if len(categorical_columns) == 0:
                                                    df_new = df_new.copy()
                                                else:
                                                    for feature in df_new.columns: 
                                                        if df_new[feature].dtype == 'object': 
                                                            print('\n')
                                                            print('feature:',feature)
                                                            print(pd.Categorical(df_new[feature].unique()))
                                                            print(pd.Categorical(df_new[feature].unique()).codes)
                                                            df_new[feature] = pd.Categorical(df_new[feature]).codes
                                                
                                                time.sleep(2)
                                                st.write("Feature Scalling...")
                                                
                                                if scaling_reqd == 'yes':     
                                                    df_new = scale_features(df_new,scaling_method)
                                                else:
                                                    df_new = df_new.copy()
                                                    
                                                X_new = df_new[selected_features]
                                                
                                                status.update(label="**:blue[Analysis Complete]**", state="complete", expanded=False)

                                            if hasattr(best_model, "predict_proba"):
                                                proba = best_model.predict_proba(X_new)[:, 1]
                                            else:
                                                proba = [0.0] * len(X_new)
                                                st.warning("⚠️ Model does not support probability prediction.")

                                            preds_binary = (np.array(proba) >= 0.5).astype(int)

                                            if 'label_map' in locals():
                                                preds_mapped = [label_map.get(p, p) for p in preds_binary]
                                            else:
                                                preds_mapped = preds_binary

                                            new_df_result = df_new.copy()
                                            new_df_result["Predicted_Score"] = proba
                                            new_df_result["Predicted_Label"] = preds_mapped
                                            st.markdown("##### 📋 Prediction Results (New Dataset)")
                                            #st.dataframe(new_df_result[[*selected_features, "Predicted_Score", "Predicted_Label"]],
                                                        #use_container_width=True)
                                            st.dataframe(new_df_result, use_container_width=True)

                                        except Exception as e:
                                            st.error(f"🚫 Prediction failed: {e}")

                        with tabs[10]:
                            
                            st.divider()
                            #with st.expander("📉 Drift Analysis on Prediction Probabilities"):
                                #drift_df = compute_drift_matrix(train_pred_proba, new_pred_proba)
                                #st.dataframe(drift_df.style.format({"drift_score": "{:.6f}"}), use_container_width=True)   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
#---------------------------------------------------------------------------------------------------------------------------------   
elif ml_type == "Regression":  

    col1, col2= st.columns((0.15,0.85))
    with col1:           
        with st.container(border=True):
            
            reg_file = st.file_uploader("📁 **:blue[Choose file (for training)]**",type=["csv", "xls", "xlsx","parquet"], accept_multiple_files=False, key="file_upload")
            if reg_file is not None:
                st.success("Data loaded successfully!")
                df = load_file(reg_file) 
                
                st.divider()
 
                df_new = None
                new_data_file = st.file_uploader("📁 **:blue[Choose file for prediction (optional)]**", type=["csv", "xlsx", "xls", "parquet"], key="new_data") 
                if new_data_file:
                    df_new = load_file(new_data_file)
                    st.success("✅ New data uploaded successfully.")
                
                st.divider()
                                
                target_variable = st.selectbox("**:blue[Choose Target Variable]**", options=["None"] + list(df.columns), key="target_variable")
                
                with st.expander("**✂️ Features for Deletion**", expanded=False):
                    optional_cols_to_delete = st.multiselect("**:blue[Feature Removal (optional)]**",
                    options=[col for col in df.columns if col != target_variable],
                    help="Choose columns to drop from dataset before proceeding.")
                if optional_cols_to_delete:
                    st.info(f"🗑️ Dropping {len(optional_cols_to_delete)} selected column(s): {optional_cols_to_delete}")
                    df = df.drop(columns=optional_cols_to_delete)
                else:
                    st.success("✅ No columns selected for deletion.")
                    
                if target_variable == "None":
                    st.warning("Please choose a target variable to proceed with the analysis.")                
                
                else:
                    st.warning("Tune or Change the **Hyperparameters**(tab shown in the top) whenever required.")   
                    with col2:

                        with st.popover("**:blue[:hammer_and_wrench: Hyperparameters]**",disabled=False, use_container_width=True,help="Tune the hyperparameters whenever required"):

                            subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
                            with subcol1:      
                                    numerical_strategies = ['mean', 'median', 'most_frequent']
                                    categorical_strategies = ['constant','most_frequent']
                                    selected_numerical_strategy = st.selectbox("**Missing value treatment : Numerical**", numerical_strategies)
                                    selected_categorical_strategy = st.selectbox("**Missing value treatment : Categorical**", categorical_strategies) 
                                    st.divider() 
                                    treatment_option = st.selectbox("**Select a outlier treatment option**", ["Cap Outliers","Drop Outliers", ])

                            with subcol2: 
                                    scaling_reqd = st.selectbox("**Requirement of scalling**", ["no", "yes"])
                                    if scaling_reqd == 'yes':                       
                                        scaling_method = st.selectbox("**Scaling method**", ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
                                    if scaling_reqd == 'no':   
                                        scaling_method = 'N/A'
                                    st.divider()
                                    vif_threshold = st.number_input("**VIF Threshold**",2.0,5.0,5.0)                        
                                    st.divider()
                                    num_features_to_select = st.slider("**Number of Independent Features**",1,len(df.columns),5)
                                    st.divider()
                                    threshold = st.number_input("**Variance Threshold**",0.0,0.1,0.05)  

                            with subcol3:                    
                                    train = st.slider("**Train Size (as %)**", 10, 90, 70, 5)
                                    test = st.slider("**Test Size (as %)**", 10, 50, 30, 5)    
                                    random_state = st.number_input("**Random State**", 0, 100, 42)
                                    n_jobs = st.number_input("**Parallel Processing (n_jobs)**", -10, 10, 1)    
                                    st.divider()
                                    selected_metric_reg = st.selectbox("**Select the metric to decide the best model for fitting**", ["R2", "MAE", "MSE", "RMSE", "MAPE"])
                                    
                            with subcol4: 
                                with st.expander("**📌 Parameters | Tree & Boosting Method**", expanded=False):                                                                                                                                                
                                    n_estimators = st.slider("**Number of Estimators**", 10, 1000, 50, 100)
                                    criterion = st.selectbox("**Criteria**", ["gini", "entropy", "log_loss"]) 
                                    max_depth = st.slider("**Max Depth**", 1, 20, 1, 10)    
                                    loss = st.selectbox("**Loss Function**", ["log_loss","exponential"]) 
                                    #min_samples_split = st.slider("**Min Samples Split**", 2, 10, 1, 2)
                                    min_samples_leaf = st.slider("**Min Samples Leaf**", 2, 10, 1, 2)
                                    learning_rate = st.number_input("**Learning rate**",0.01,1.00,0.01,0.10, key ='learning_rate')
                                    #gamma =st.number_input("**min_split_loss**",0.00,1.00,0.05,0.1)
                                    #lamda = st.slider("**L2 reg**", 0, 100, 1, 10)
                                    alpha = st.slider("**L1 reg**", 0, 100, 1, 10)
                                    eval_metric = st.selectbox("**Evaluation Metrics**", ["logloss", "auc", "aucpr","ndcg","map"])

                                st.divider()
                                kernel = st.selectbox("**Kernel**", ["linear", "poly", "rbf", "sigmoid"])
                                gamma = st.selectbox("**Gamma**", ["scale", "auto"])                                      

                            with subcol5: 
                                    k = st.slider("**Number of K folds**", min_value=3, max_value=10, step=1, value=5)  
                                    st.divider()
                                    search_method = st.selectbox("**🔍 Select Search Method**", ["Grid Search", "Randomized Search"])
                                    n_iter = st.slider("**Iterations for RandomizedSearchCV**", 1, 20, 10) if search_method == "Randomized Search" else None
                                    st.divider()
                                    scoring_reg = st.selectbox("**Scoring (Regression)**", ["neg_root_mean_squared_error", "-", "-", "-"])         
                        
                        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        tabs = st.tabs(["**📊 Overview**","**📈 Visualizations**","**🔧 Preprocessing**","**✅ Assumptions**","**⚖️ Comparison**","**🎯 Importance**","**🔄 Validation**","**📈 Graph**","**📋 Results**","**🎲 Prediction**", "**⚠︎ Drift**"])
                        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        with tabs[0]:
                            
                            unwanted_substrings = ['unnamed', 'nan', 'deleted']
                            cols_to_delete = [col for col in df.columns if any(sub in col.lower() for sub in unwanted_substrings)]
                            if cols_to_delete:
                                st.warning(f"🗑️ {len(cols_to_delete)} column(s) deleted.")
                            else:
                                st.info("✅ No unwanted columns found. Nothing deleted after importing.")
                            df= df.drop(columns=cols_to_delete)
                           
                            st.info("✅ Showing Top 3 rows for reference.") 
                            st.dataframe(df.head(3))
                            
                            st.divider()
                            
                            with st.container(border=True):

                                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

                                col1.metric('**input values (rows)**', df.shape[0], help='number of rows')
                                col2.metric('**variables (columns)**', df.shape[1], help='number of columns')     
                                col3.metric('**numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
                                col4.metric('**categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')
                                col5.metric('**Missing values**', df.isnull().sum().sum(), help='Total missing values in the dataset')
                                col6.metric('**Target Variable**', target_variable, help='Selected target variable')  
                                
                            with st.container(border=True):
                                
                                df_describe_table = df.describe().T.reset_index().rename(columns={'index': 'Feature'})
                                st.markdown("##### 📊 Descriptive Statistics")
                                st.dataframe(df_describe_table)

                        with tabs[1]:
                            
                            cat_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
                            num_vars = df.select_dtypes(include=['number']).columns.tolist()
                            
                            with st.container(border=True):
                                if cat_vars:
                                    st.success(f"📋 Categorical Variables Found: {len(cat_vars)}")
                                else:
                                    st.warning("⚠️ No categorical variables found.")

                                if cat_vars:
                                    for i in range(0, len(cat_vars), 3):
                                        cols = st.columns(3)
                                        for j, col_name in enumerate(cat_vars[i:i+3]):
                                            with cols[j]:
                                                fig, ax = plt.subplots(figsize=(4, 3))
                                                df[col_name].value_counts().plot(kind='bar', ax=ax, color='skyblue')
                                                ax.set_title(f"{col_name}", fontsize=10)
                                                ax.set_ylabel("Count", fontsize=9)
                                                ax.set_xlabel("")
                                                ax.tick_params(axis='x', rotation=45, labelsize=8)
                                                ax.tick_params(axis='y', labelsize=8)
                                                st.pyplot(fig,use_container_width=True)

                            with st.container(border=True):
                                if num_vars:
                                    st.success(f" 📈 Numerical Variables Found: {len(num_vars)}")
                                else:
                                    st.warning("⚠️ No numerical variables found.")

                                if num_vars:
                                    for i in range(0, len(num_vars), 3):
                                        cols = st.columns(3)
                                        for j, col_name in enumerate(num_vars[i:i+3]):
                                            with cols[j]:
                                                st.markdown(f"**{col_name}**")
                                                skew_val = df[col_name].skew()
                                                skew_tag = (
                                                    "🟩 Symmetric" if abs(skew_val) < 0.5 else
                                                    "🟧 Moderate skew" if abs(skew_val) < 1 else
                                                    "🟥 Highly skewed"
                                                )
                                                st.info(f"Skewness: {skew_val:.2f} — {skew_tag}")

                                                fig_box, ax_box = plt.subplots(figsize=(4,2))
                                                sns.boxplot(y=df[col_name], ax=ax_box, color='lightcoral')
                                                ax_box.set_title("Box Plot",fontsize=8)
                                                ax_box.set_ylabel("", fontsize=8)
                                                ax_box.set_xlabel("", fontsize=8)
                                                ax_box.tick_params(axis='y', labelsize=8)
                                                st.pyplot(fig_box,use_container_width=True)

                                                fig_hist, ax_hist = plt.subplots(figsize=(4,2))
                                                sns.histplot(df[col_name], kde=True, ax=ax_hist, color='steelblue')
                                                ax_hist.set_title("Histogram", fontsize=8)
                                                ax_hist.set_xlabel("" ,fontsize=8)
                                                ax_hist.set_ylabel("", fontsize=8)
                                                ax_hist.tick_params(axis='x', labelsize=8)
                                                ax_hist.tick_params(axis='y', labelsize=8)
                                                st.pyplot(fig_hist,use_container_width=True)
                                                
                                        st.markdown('---')

                            with st.container(border=True):
                                    
                                    numeric_df = df.select_dtypes(include=['number'])
                                    cmap = sns.diverging_palette(220,20,as_cmap=True)
                                    corrmat = numeric_df.corr()
                                    fig, ax = plt.subplots(figsize=(20,20))
                                    sns.heatmap(corrmat, annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax)
                                    ax.set_title("Correlation Heatmap", fontsize=14)
                                    plt.xticks(rotation=45, ha="right", fontsize=10)
                                    plt.yticks(rotation=0, fontsize=10)
                                    st.pyplot(fig, use_container_width=True)

                                    df_cor = df.copy()
                                    if target_variable in df_cor.columns:
                                        if target_variable in num_vars:
                                            num_vars.remove(target_variable)
                                        if not pd.api.types.is_numeric_dtype(df_cor[target_variable]):
                                            le = LabelEncoder()
                                            df_cor[target_variable] = le.fit_transform(df_cor[target_variable])
                                        numeric_cols = df_cor[num_vars + [target_variable]].select_dtypes(include=['number'])
                                        target_corr = numeric_cols.corr()[target_variable].drop(target_variable).reset_index()
                                        target_corr.columns = ["Variable", f"Correlation with {target_variable}"]
                                        target_corr = target_corr.sort_values(by=f"Correlation with {target_variable}", ascending=False)
                                        
                                        st.markdown(f"##### 🎯 Correlation of Variables with **{target_variable}**")
                                        st.dataframe(target_corr.reset_index(drop=True), hide_index=True)
                                    else:
                                        st.warning(f"⚠️ Target variable `{target_variable}` not found in the dataset.")
                                                                                                        
                        with tabs[2]:
                            
                            with st.container(border=True):
                                
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
                                            st.download_button("📥 Download data as .csv", cleaned_df.to_csv(index=False), file_name="treated_data.csv")

                            with st.container(border=True):

                                if st.checkbox("Show Duplicate Values"):
                                        if missing_values.empty:
                                            st.table(df[df.duplicated()].head(2))
                                        else:
                                            st.table(cleaned_df[cleaned_df.duplicated()].head(2))

                            with st.container(border=True):

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
                                        st.dataframe(outliers,hide_index=True)
                    
                                with col2:
                                    
                                    if treatment_option == "Drop Outliers":
                                        df = df[~outliers['Column'].isin(outliers[outliers['Number of Outliers'] > 0]['Column'])]
                                        st.success("Outliers dropped. Preview of the cleaned dataset:")
                                        st.table(df.head())
                                    
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
                                            st.dataframe(df,hide_index=True)     

                            with st.container(border=True):

                                    categorical_columns = df.select_dtypes(include=['object']).columns
                                    if len(categorical_columns) == 0:
                                        st.warning("There are no categorical variables in the dataset.Proceed with the original DataFrame")
                                        df = df.copy()
                                    else:
                                        for feature in df.columns: 
                                            if df[feature].dtype == 'object': 
                                                print('\n')
                                                print('feature:',feature)
                                                print(pd.Categorical(df[feature].unique()))
                                                print(pd.Categorical(df[feature].unique()).codes)
                                                df[feature] = pd.Categorical(df[feature]).codes
                                        st.success("**Categorical variables are encoded**")
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(label="📥 Download Encoded Data (for review)", data=csv, file_name='encoded_data.csv', mime='text/csv')

                            with st.container(border=True):
                                
                                    if scaling_reqd == 'yes':     
                                        df = scale_features(df,scaling_method)
                                        st.success("**Data is scaled for further treatment**")
                                        csv = df.to_csv(index=False).encode('utf-8')
                                        st.download_button(label="📥 Download Scaled Data (for review)", data=csv, file_name='scaled_data.csv', mime='text/csv')
                                    else:
                                        st.warning("Data is not scaled, orginal data is considered for further treatment | If Scalling required, change the options in the drop-down menu from **Hyperparameter** tab in the top.")
                                    #st.dataframe(df.head()) 

                            with st.container(border=True):                                        

                                X = df.drop(columns=target_variable)
                                y = df[target_variable]
                                st.markdown(f"""
                                **Original number of features**: **{X.shape[1]}**

                                🧪 The following feature selection methods are applied sequentially to identify the most optimal features:
                                - Variance Threshold
                                - SelectKBest 
                                - VIF (Variance Influence Factor)
                                """)
                                st.divider()
                                col1, col2, col3 = st.columns(3)
                                with col1:

                                        st.markdown("**Feature Selection Method : VarianceThreshold**")

                                        X = df.drop(columns=target_variable)
                                        selector = VarianceThreshold(threshold=threshold)
                                        X_varth = selector.fit_transform(X)
                                        selected_varth_indices = selector.get_support(indices=True)
                                        selected_varth_features = X.columns[selected_varth_indices]
                                        feature_variances = selector.variances_
                                        variance_df = pd.DataFrame({
                                            "Feature": X.columns,
                                            "Variance": feature_variances,
                                            "Status": ["✅ Kept" if i in selected_varth_indices else "❌ Dropped" for i in range(len(X.columns))]
                                        }).sort_values(by="Variance", ascending=True)
                                        st.write(f"✅ Number of features before Variance Threshold: **{X.shape[1]}**")
                                        st.dataframe(variance_df, hide_index=True)

                                        # Remarks
                                        dropped_features = variance_df[variance_df["Status"] == "❌ Dropped"]
                                        st.markdown(f"""
                                        ℹ️ **Remarks:**  
                                        - Threshold for variance filtering: **{threshold}**  
                                        - Total features before filtering: **{X.shape[1]}**  
                                        - Features dropped due to low variance: **{len(dropped_features)}**

                                        These features had variance below the set threshold and were considered non-informative.
                                        """)

                                with col2:
                  
                                        st.markdown("**Feature Selection Method : Selectkbest**")          

                                        st.markdown(f"✅Number of features before SelectKBest method: **{len(selected_varth_features)}**")
                                        X_kbest_input = df[selected_varth_features].copy()
                                        y = df[target_variable]
                                        num_features_to_select = X_kbest_input.shape[1]
                                        f_selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
                                        X_f_selected = f_selector.fit_transform(X_kbest_input, y)
                                        f_scores = f_selector.scores_
                                        f_pvalues = f_selector.pvalues_
                                        mi_scores = mutual_info_regression(X_kbest_input, y, random_state=42)
                                        selection_df = pd.DataFrame({
                                            "Feature": selected_varth_features,
                                            "F-Score": f_scores,
                                            "p-value": f_pvalues,
                                            "Mutual Info Score": mi_scores
                                        }).sort_values(by="F-Score", ascending=False)
                                        st.dataframe(selection_df, hide_index=True)
                                        pval_threshold = 0.05
                                        filtered_selection_df = selection_df[selection_df["p-value"] < pval_threshold].copy()
                                        st.markdown(f"""
                                        ℹ️ **Remarks:**  
                                        - Total features before p-value filtering: **{len(selection_df)}**  
                                        - Features with **p-value < {pval_threshold}** retained: **{len(filtered_selection_df)}**  
                                        - These are passed to the next step (**VIF filtering**).
                                        """)
                                        st.divider()
                                        st.markdown("**Filtered Features (Significant by F-test):**")
                                        st.dataframe(filtered_selection_df, hide_index=True)

                                with col3:
                                                                            
                                        st.markdown("**Feature Selection Method : VIF**")  

                                        selected_kbest_features = filtered_selection_df["Feature"].tolist()
                                        X_vif_input = df[selected_kbest_features].copy()
                                        filtered_vif_df, final_vif_table = iterative_vif_filtering(X_vif_input, threshold=vif_threshold)
                                        st.write(f"✅ Final number of features after VIF filtering: {filtered_vif_df.shape[1]}")
                                        st.dataframe(final_vif_table, hide_index=True)

                            #selected_features = filtered_vif_df.copy()  

                        with tabs[3]: 
                            
                            selected_features = filtered_vif_df.columns.tolist()
                            X = filtered_vif_df.copy()
                            y = df[target_variable]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

                            lr_model = LinearRegression()
                            lr_model.fit(X_train, y_train)
                            y_pred_lr = lr_model.predict(X_test)

                            with st.container(border=True):
                                st.markdown(f"""
                                ✅ **Final dataset prepared for modeling**  
                                - Number of selected features: **{len(selected_features)}**  
                                - Train set size: **{X_train.shape[0]} rows**  
                                - Test set size: **{X_test.shape[0]} rows**
                                """) 
                                
                            with st.container(border=True):
                                #check_linear_regression_assumptions(y_test, y_pred_lr, X_test, lr_model)   
                                df_assumption_check = check_linear_regression_assumptions(y_test, y_pred_lr, X_test, lr_model)
                                
                        with tabs[4]: 
                                            
                            with st.spinner("Setting up and comparing models..."):

                                        results = []
                                        linear_reg_valid = df_assumption_check["Status"].eq("Reject").sum() == 0  # all must be Accept
                                        valid_regressors = regressors.copy()
                                        if not linear_reg_valid and "Linear Regression" in valid_regressors:
                                                    del valid_regressors["Linear Regression"]
                                                    st.warning("❌ Linear Regression excluded from comparison due to unmet assumptions.")
                                                    
                                        for name, model in valid_regressors.items():
                                                    model.fit(X_train, y_train)
                                                    y_pred = model.predict(X_test)
                                                    mae, mse, rmse, r2, rmsle, mape_value = calculate_metrics(y_test, y_pred)
                                                    results.append({"Model": name,
                                                                    "MAE": round(mae, 2),
                                                                    "MSE": round(mse, 2),
                                                                    "RMSE": round(rmse, 2),
                                                                    "R2": round(r2, 2),
                                                                    #"RMSLE": round(rmsle, 2) if rmsle else "N/A",
                                                                    "MAPE": round(mape_value, 2)})
                                        results_df = pd.DataFrame(results)

                                        results_train = []                                        
                                        for name, model in valid_regressors.items():
                                                    model.fit(X_train, y_train)
                                                    y_pred_train = model.predict(X_train)
                                                    mae, mse, rmse, r2, rmsle, mape_value = calculate_metrics(y_train, y_pred_train)
                                                    results_train.append({"Model": name,
                                                                    "MAE": round(mae, 2),
                                                                    "MSE": round(mse, 2),
                                                                    "RMSE": round(rmse, 2),
                                                                    "R2": round(r2, 2),
                                                                    #"RMSLE": round(rmsle, 2) if rmsle else "N/A",
                                                                    "MAPE": round(mape_value, 2)})
                                        results_df_train = pd.DataFrame(results_train)                                        
                                        
                                        with st.container(border=True):                                    
                                    
                                            col1, col2= st.columns(2)
                                            with col1:
                                                st.markdown("##### Model Metrices Comparison | Train Dataset")
                                                st.dataframe(results_df_train, hide_index=True)
                                        
                                            with col2:
                                                st.markdown("##### Model Metrices Comparison | Test Dataset") 
                                                st.dataframe(results_df, hide_index=True)                                            
                                            
                                        with st.expander("**🏆 Best Model per Metric**", expanded=False): 
                                         
                                            col1, col2= st.columns(2)
                                            with col1: 
                                                st.markdown("##### Train Dataset")                                            
                                                metric_cols = [col for col in results_df.columns if col != "Model"]
                                                best_models_per_metric = {
                                                    metric: results_df.loc[results_df[metric].idxmax(), "Model"]
                                                    for metric in metric_cols}
                                                summary_df = pd.DataFrame({"Metric": list(best_models_per_metric.keys()),"Best Model": list(best_models_per_metric.values())})
                                                st.dataframe(summary_df, hide_index=True) 

                                            with col2:
                                                st.markdown("##### Test Dataset")                                              
                                                metric_cols = [col for col in results_df_train.columns if col != "Model"]
                                                best_models_per_metric = {
                                                    metric: results_df_train.loc[results_df_train[metric].idxmax(), "Model"]
                                                    for metric in metric_cols}
                                                summary_df_train = pd.DataFrame({"Metric": list(best_models_per_metric.keys()),"Best Model": list(best_models_per_metric.values())})
                                                st.dataframe(summary_df_train, hide_index=True) 
                                                
                                        best_model_reg = best_models_per_metric[selected_metric_reg]
                                        best_model = regressors[best_model_reg]
                                        
                                        y_pred_best_train  = best_model.predict(X_train)
                                        residuals_train = y_train - y_pred_best_train                                         
                                        y_pred_best = best_model.predict(X_test)
                                        residuals = y_test - y_pred_best              
                                          
                                        with st.container(border=True):
                                            
                                            metrics = ["MAE", "MSE", "RMSE", "R2", "MAPE"]
                                            metrics_df_train = results_df_train.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
                                            metrics_df_test = results_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")

                                            for metric in metrics:
                                                st.markdown(f"##### 📊 **{metric}** — Train vs Test Comparison")
                                                col1, col2 = st.columns(2)

                                                train_vals = metrics_df_train[metrics_df_train["Metric"] == metric].set_index("Model")["Value"]
                                                test_vals = metrics_df_test[metrics_df_test["Metric"] == metric].set_index("Model")["Value"]
                                                delta_vals = (test_vals - train_vals).round(2)

                                                with col1:
                                                    st.markdown("##### 🟦 Train")
                                                    fig, ax = plt.subplots(figsize=(6, 2.5))
                                                    bars = sns.barplot(x="Model", y="Value",data=metrics_df_train[metrics_df_train["Metric"] == metric],color="steelblue", ax=ax)
                                                    for bar in bars.patches:
                                                        height = bar.get_height()
                                                        bars.annotate(f"{height:.2f}",(bar.get_x() + bar.get_width() / 2, height),ha='center', va='bottom', fontsize=8, color='black')
                                                    ax.set_title(f"Train: {metric}", fontsize=8)
                                                    ax.set_ylabel(metric, fontsize=8)
                                                    ax.set_xlabel("", fontsize=8)
                                                    ax.tick_params(axis='x', rotation=90, labelsize=8)
                                                    st.pyplot(fig, use_container_width=True)

                                                with col2:
                                                    st.markdown("##### 🟨 Test + Δ (Delta)")
                                                    fig, ax = plt.subplots(figsize=(6, 2.5))
                                                    bars = sns.barplot(x="Model", y="Value",data=metrics_df_test[metrics_df_test["Metric"] == metric],color="darkorange", ax=ax)
                                                    for bar in bars.patches:
                                                        height = bar.get_height()
                                                        model_name = bar.get_x() + bar.get_width() / 2
                                                        bars.annotate(f"{height:.2f}",(model_name, height),ha='center', va='bottom', fontsize=8, color='black')
                                                    
                                                    for i, bar in enumerate(bars.patches):
                                                        model = metrics_df_test[metrics_df_test["Metric"] == metric]["Model"].unique()[i]
                                                        delta = delta_vals.get(model, 0.0)
                                                        delta_str = f"🔺 +{delta:.2f}" if delta > 0 else f"🔻 {delta:.2f}"
                                                        ax.annotate(delta_str,(bar.get_x() + bar.get_width() / 2, 0),ha='center', va='top', fontsize=8, color='green' if delta <= 0 else 'red')

                                                    ax.set_title(f"Test: {metric}", fontsize=8)
                                                    ax.set_ylabel(metric, fontsize=8)
                                                    ax.set_xlabel("", fontsize=8)
                                                    ax.tick_params(axis='x', rotation=90, labelsize=8)
                                                    st.pyplot(fig, use_container_width=True)

                        with tabs[5]:

                            if best_model_reg == "Linear Regression":
                                importance_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': best_model.coef_[0]})
                                importance_df['Importance'] = abs(importance_df['Coefficient'])
                                importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
                                importance_df = importance_df.sort_values(by='Importance', ascending=False)

                            else:
                                importance_df = pd.DataFrame({"Feature": selected_features,"Importance": best_model.feature_importances_})
                                importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
                                importance_df = importance_df.sort_values(by='Importance', ascending=False)                           
                            
                            col1, col2 = st.columns((0.25, 0.75))
                            with col1:
                                with st.container(border=True):

                                    st.dataframe(importance_df, hide_index=True)

                            with col2:
                                with st.container(border=True):

                                    plot_data_imp = [go.Bar(x=importance_df['Feature'], y=importance_df['Importance'])]
                                    plot_layout_imp = go.Layout(xaxis={"title": "Feature"},yaxis={"title": "Importance"},title="Feature Importance")
                                    fig = go.Figure(data=plot_data_imp, layout=plot_layout_imp)
                                    st.plotly_chart(fig, use_container_width=True)

                            with st.container(border=True):
                            
                                top_n = min(5, len(importance_df))
                                top_features_text = "\n".join([
                                        f"{i+1}. {importance_df.iloc[i]['Feature']} ({importance_df.iloc[i]['Percentage']:.2f}%)"
                                        for i in range(top_n)])
                                    
                                with st.expander("📌 Top Features Summary", expanded=True):
                                        if top_n > 0:
                                            st.info(f"**Top Features based on Importance:**\n{top_features_text}")
                                        else:
                                            st.warning("⚠️ No features available to display.")

                            with st.container(border=True):
                                
                                    if best_model_reg != "Linear Regression":
                                        st.markdown("##### 🌐 SHAP Explanation (Top Features)")
                                        try:
                                            explainer = shap.Explainer(best_model, X)
                                            shap_values = explainer(X)
                                            fig, ax = plt.subplots(figsize=(20,5))
                                            shap.plots.bar(shap_values, max_display=10, show=False)
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                        except Exception as e:
                                            st.warning(f"⚠️ SHAP plot could not be generated: {e}") 

                        with tabs[6]:   
                                                                      
                            col1, col2 = st.columns(2)
                            with col1:
                        
                                with st.container(border=True):
                                
                                    kf = KFold(n_splits=k, shuffle=True, random_state=42)
                                    scores = cross_val_score(best_model, X_train, y_train, cv=kf, scoring=scoring_reg)
                                    mean_score = np.mean(scores)
                                    std_score = np.std(scores)
                                    results_val = pd.DataFrame({"Fold": list(range(1, k + 1)),"Score": [round(s, 4) for s in scores]})
                                    st.dataframe(results_val, hide_index=True)
                                    st.divider()
                                    metric_name = scoring_reg.replace("neg_", "").upper()
                                    positive_mean_score = abs(mean_score)
                                    approx_rmse = np.sqrt(positive_mean_score)
    
                            with col2:
                        
                                with st.container(border=True):
                                    
                                    st.markdown("##### 📊 Remarks")

                                    explanation = f"""
                                    - The metric used was: **{metric_name}**
                                    - In scikit-learn, negative values (like -MSE) mean the **lower the better**.
                                    - The **positive equivalent** of the average score is: **{positive_mean_score:.4f}**
                                    - Approximate **RMSE** (Root Mean Squared Error): **{approx_rmse:.4f}**
                                    - Standard deviation across folds: **{std_score:.4f}** — {'stable' if std_score < 0.5 else 'variable'} performance.
                                    > ⚠️ Whether this is a good result depends on the scale of your target variable.
                                    """
                                    st.markdown(explanation)

                            with st.container(border=True):
                                    
                                    st.markdown("##### 📊 Performance Summary")
                                    st.info(
                                        f"**{best_model_reg}** achieved an average **{mean_score:.4f}** "
                                        f"(± **{std_score:.4f}**) using **{k}-Fold Cross-Validation**."
                                    )
                                                                                
                        with tabs[7]:    
                            
                                    col1, col2 = st.columns(2)  
                                    with col1:
                                        with st.container(border=True):                
                                                      
                                            plt.figure(figsize=(8, 3))
                                            sns.residplot(x=y_pred_best, y=residuals, lowess=True)
                                            plt.title(f"Residual Plot", fontsize=8)
                                            plt.xlabel('Predicted', fontsize=8)
                                            plt.ylabel('Residuals', fontsize=8)
                                            st.pyplot(plt,use_container_width=True)
    
                                            plt.figure(figsize=(8, 3))
                                            sns.scatterplot(x=y_test, y=y_pred_best)
                                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
                                            plt.title(f"Prediction Error Plot", fontsize=8)
                                            plt.xlabel('Actual', fontsize=8)
                                            plt.ylabel('Predicted', fontsize=8)
                                            st.pyplot(plt,use_container_width=True) 

                                    with col2:
                                        with st.container(border=True): 

                                            plot_learning_curve(best_model, X_train, y_train)  

                                            #param_name = 'alpha'  
                                            #param_range = np.logspace(-3, 3, 10)
                                            #plot_validation_curve(best_model, X_train, y_train, param_name, param_range)   

                        with tabs[8]:  

                                best_metrics=results_df.loc[results_df["Model"] == best_model_reg].iloc[0].to_dict()
                                final_results_df = pd.DataFrame({"Metric": ["Type of Problem",
                                                                                "Target Variable",
                                                                                "Scaling Method", 
                                                                                "Best Algorithm", 
                                                                                "MAE", 
                                                                                "MSE", 
                                                                                "RMSE", 
                                                                                "R2", 
                                                                                "MAPE", 
                                                                                #"Best Feature(s)",
                                                                                ],
                                                                    "Value": [ml_type,target_variable,
                                                                              scaling_method, 
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

                                        with st.container(border=True):                          
                                            st.dataframe(final_results_df, hide_index=True, use_container_width=True)

                                with col2:
                                        with st.container(border=True):
                             
                                            best_model.fit(X_train, y_train)
                                            y_pred_best = best_model.predict(X_test)
                                            X_test_results_reg = X_test.copy()  
                                            X_test_results_reg["Actual"] = y_test 
                                            X_test_results_reg["Predicted"] = y_pred_best 
                                            st.dataframe(X_test_results_reg, use_container_width=True)
                                            st.download_button(label="📥 Download data as .csv",data=X_test_results_reg.to_csv(index=False),file_name="regression_predictions.csv",mime="text/csv") 

                        with tabs[9]:   
                            
                                    if new_data_file and 'importance_df' in locals() and 'best_model' in locals():
                                        
                                        try:
                                            
                                            st.markdown("##### New Dataset (for prediction)")
                                            unwanted_substrings = ['unnamed', '0', 'nan', 'deleted']
                                            cols_to_delete = [col for col in df_new.columns if any(sub in col.lower() for sub in unwanted_substrings)]
                                            if cols_to_delete:
                                                st.warning(f"🗑️ {len(cols_to_delete)} column(s) deleted.")
                                            else:
                                                st.info("✅ No unwanted columns found. Nothing deleted after importing.")
                                                
                                            df_new= df_new.drop(columns=cols_to_delete)
                                            st.table(df_new.head(2))
                                            
                                            with st.status("**:blue[Collecting Information & Analyzing...]**", expanded=False) as status:
                                                
                                                st.write("Feature Cleaning...")
                                                
                                                missing_values = check_missing_values(df_new)
                                                cleaned_df = handle_numerical_missing_values(df_new, numerical_strategy='mean')
                                                cleaned_df = handle_categorical_missing_values(cleaned_df, categorical_strategy='most_frequent')   
                                                if missing_values.empty:
                                                    df_new = df_new.copy()
                                                else:
                                                    df_new = cleaned_df.copy()
                                                    
                                                outliers = check_outliers(df_new)
                                                df_new = df_new.copy()
                                                for column in outliers['Column'].unique():
                                                    Q1 = df_new[column].quantile(0.25)
                                                    Q3 = df_new[column].quantile(0.75)
                                                    IQR = Q3 - Q1
                                                    threshold = 1.5
                                                    df_new[column] = np.where(df_new[column] < Q1 - threshold * IQR, Q1 - threshold * IQR, df_new[column])
                                                    df_new[column] = np.where(df_new[column] > Q3 + threshold * IQR, Q3 + threshold * IQR, df_new[column])

                                                time.sleep(2)
                                                st.write("Feature Encoding...")
                                                
                                                categorical_columns = df_new.select_dtypes(include=['object']).columns
                                                if len(categorical_columns) == 0:
                                                    df_new = df_new.copy()
                                                else:
                                                    for feature in df_new.columns: 
                                                        if df_new[feature].dtype == 'object': 
                                                            print('\n')
                                                            print('feature:',feature)
                                                            print(pd.Categorical(df_new[feature].unique()))
                                                            print(pd.Categorical(df_new[feature].unique()).codes)
                                                            df_new[feature] = pd.Categorical(df_new[feature]).codes
                                                
                                                time.sleep(2)
                                                st.write("Feature Scalling...")
                                                
                                                if scaling_reqd == 'yes':     
                                                    df_new = scale_features(df_new,scaling_method)
                                                else:
                                                    df_new = df_new.copy()
                                                    
                                                X_new = df_new[selected_features]
                                                
                                                status.update(label="**:blue[Analysis Complete]**", state="complete", expanded=False)

                                        except Exception as e:
                                            st.error(f"🚫 Prediction failed: {e}")

                        with tabs[10]:
                            
                            st.divider()     
                                                                                                                                                                                                                                                                                                                                                                                                                       
#---------------------------------------------------------------------------------------------------------------------------------    
elif ml_type == "Clustering": 
        
    col1, col2= st.columns((0.15,0.85))
    with col1:           
        with st.container(border=True):
            
            reg_file = st.file_uploader("📁 **:blue[Choose file (for training)]**",type=["csv", "xls", "xlsx", "parquet"], accept_multiple_files=False, key="file_upload")
            if reg_file is not None:
                st.success("Data loaded successfully!")
                df = load_file(reg_file) 
                             
                with col2:

                        with st.popover("**:blue[:hammer_and_wrench: Hyperparameters]**",disabled=False, use_container_width=True,help="Tune the hyperparameters whenever required"):

                            subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
                            with subcol1:      
                                    numerical_strategies = ['mean', 'median', 'most_frequent']
                                    categorical_strategies = ['constant','most_frequent']
                                    selected_numerical_strategy = st.selectbox("**Missing value treatment : Numerical**", numerical_strategies)
                                    selected_categorical_strategy = st.selectbox("**Missing value treatment : Categorical**", categorical_strategies) 
                                    st.divider() 
                                    treatment_option = st.selectbox("**Select a outlier treatment option**", ["Cap Outliers","Drop Outliers", ])

                            with subcol2: 
                                    scaling_reqd = st.selectbox("**Requirement of scalling**", ["no", "yes"])
                                    if scaling_reqd == 'yes':                       
                                        scaling_method = st.selectbox("**Scaling method**", ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
                                    if scaling_reqd == 'no':   
                                        scaling_method = 'N/A'
                                    st.divider()
                                    vif_threshold = st.number_input("**VIF Threshold**", 2.0,5.0,5.0)                        
                                    st.divider()
                                    num_features_to_select = st.slider("**Number of Independent Features**",1,len(df.columns),5)
                                    st.divider()
                                    threshold = st.number_input("**Variance Threshold**",0.0,0.1,0.05)      

                            with subcol3:                    
                                    train = st.slider("**Train Size (as %)**", 10,90,70,5)
                                    test = st.slider("**Test Size (as %)**", 10,50,30,5)    
                                    random_state = st.number_input("**Random State**", 0,100,42)
                                    n_jobs = st.number_input("**Parallel Processing (n_jobs)**", -10,10,1)    
                                    
                            with subcol4: 
                                    n_components = st.slider("**Number of PCA Components**",2,20,5,1)
                                    st.divider()
                                    n_clusters_final = st.slider("**Select number of clusters**",1,10,4)
 
                            with subcol5: 
                                    st.divider()
                              
                        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        tabs = st.tabs(["**📊 Overview**","**📈 Visualizations**","**🔧 Preprocessing**","**✅ Dimensionality**","**⚖️ Distribution**","**🔄 Validation**","**📈 Graph**","**📋 Results**",])
                        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        with tabs[0]:
                            
                            unwanted_substrings = ['unnamed', 'nan', 'deleted']
                            cols_to_delete = [col for col in df.columns if any(sub in col.lower() for sub in unwanted_substrings)]
                            if cols_to_delete:
                                st.warning(f"🗑️ {len(cols_to_delete)} column(s) deleted.")
                            else:
                                st.info("✅ No unwanted columns found. Nothing deleted after importing.")
                            df= df.drop(columns=cols_to_delete)
                            
                            st.info("✅ Showing Top 3 rows for reference.") 
                            st.dataframe(df.head(3))
                            df1 = df.copy()
                            
                            st.divider()
                            
                            with st.container(border=True):

                                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

                                col1.metric('**input values (rows)**', df.shape[0], help='number of rows')
                                col2.metric('**variables (columns)**', df.shape[1], help='number of columns')     
                                col3.metric('**numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
                                col4.metric('**categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')
                                col5.metric('**Missing values**', df.isnull().sum().sum(), help='Total missing values in the dataset')
                                
                            with st.container(border=True):
                                
                                df_describe_table = df.describe().T.reset_index().rename(columns={'index': 'Feature'})
                                st.markdown("##### 📊 Descriptive Statistics")
                                st.dataframe(df_describe_table)                       

                        with tabs[1]:
                            
                            cat_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
                            num_vars = df.select_dtypes(include=['number']).columns.tolist()
                            
                            with st.container(border=True):
                                if cat_vars:
                                    st.success(f"**📋 Categorical Variables Found: {len(cat_vars)}**")
                                else:
                                    st.warning("**⚠️ No categorical variables found.**")

                                if cat_vars:
                                    for i in range(0, len(cat_vars), 3):
                                        cols = st.columns(3)
                                        for j, col_name in enumerate(cat_vars[i:i+3]):
                                            with cols[j]:
                                                fig, ax = plt.subplots(figsize=(5,2.5))
                                                df[col_name].value_counts().plot(kind='bar', ax=ax, color='skyblue')
                                                ax.set_title(f"{col_name}", fontsize=10)
                                                ax.set_ylabel("Count", fontsize=9)
                                                ax.set_xlabel("")
                                                ax.tick_params(axis='x', rotation=45, labelsize=8)
                                                ax.tick_params(axis='y', labelsize=8)
                                                st.pyplot(fig,use_container_width=True)

                            with st.container(border=True):
                                if num_vars:
                                    st.success(f"**📈 Numerical Variables Found: {len(num_vars)}**")
                                else:
                                    st.warning("**⚠️ No numerical variables found.**")

                                if num_vars:
                                    for i in range(0, len(num_vars), 3):
                                        cols = st.columns(3)
                                        for j, col_name in enumerate(num_vars[i:i+3]):
                                            with cols[j]:
                                                st.markdown(f"**{col_name}**")
                                                skew_val = df[col_name].skew()
                                                skew_tag = (
                                                    "🟩 Symmetric" if abs(skew_val) < 0.5 else
                                                    "🟧 Moderate skew" if abs(skew_val) < 1 else
                                                    "🟥 Highly skewed"
                                                )
                                                st.info(f"Skewness: {skew_val:.2f} — {skew_tag}")

                                                fig_box, ax_box = plt.subplots(figsize=(4,2))
                                                sns.boxplot(y=df[col_name], ax=ax_box, color='lightcoral')
                                                ax_box.set_title("Box Plot",fontsize=8)
                                                ax_box.set_ylabel("", fontsize=8)
                                                ax_box.set_xlabel("", fontsize=8)
                                                ax_box.tick_params(axis='y', labelsize=8)
                                                st.pyplot(fig_box,use_container_width=True)

                                                fig_hist, ax_hist = plt.subplots(figsize=(4,2))
                                                sns.histplot(df[col_name], kde=True, ax=ax_hist, color='steelblue')
                                                ax_hist.set_title("Histogram", fontsize=8)
                                                ax_hist.set_xlabel("" ,fontsize=8)
                                                ax_hist.set_ylabel("", fontsize=8)
                                                ax_hist.tick_params(axis='x', labelsize=8)
                                                ax_hist.tick_params(axis='y', labelsize=8)
                                                st.pyplot(fig_hist,use_container_width=True)
                                                
                                        st.markdown('---')                               

                            with st.container(border=True):
                                    
                                    df_c = df.copy()
                                    numeric_df = df_c.select_dtypes(include=['number'])
                                    cmap = sns.diverging_palette(220,20,as_cmap=True)
                                    corrmat = numeric_df.corr()
                                    fig, ax = plt.subplots(figsize=(20,20))
                                    sns.heatmap(corrmat, annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax)
                                    ax.set_title("Correlation Heatmap", fontsize=14)
                                    plt.xticks(rotation=45, ha="right", fontsize=10)
                                    plt.yticks(rotation=0, fontsize=10)
                                    st.pyplot(fig, use_container_width=True)
                                                                          
                        with tabs[2]:
                            
                            with st.container(border=True):
                                
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
                                            st.download_button("📥 Download Treated Data (.csv)", cleaned_df.to_csv(index=False), file_name="treated_data.csv")

                            with st.container(border=True):

                                if st.checkbox("Show Duplicate Values"):
                                        if missing_values.empty:
                                            st.table(df[df.duplicated()].head(2))
                                        else:
                                            st.table(cleaned_df[cleaned_df.duplicated()].head(2))

                            with st.container(border=True):

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
                                        st.dataframe(outliers,hide_index=True)
                    
                                with col2:
                                    
                                    if treatment_option == "Drop Outliers":
                                        df = df[~outliers['Column'].isin(outliers[outliers['Number of Outliers'] > 0]['Column'])]
                                        st.success("Outliers dropped. Preview of the cleaned dataset:")
                                        st.table(df.head())
                                    
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
                                            st.dataframe(df,hide_index=True)     

                            with st.container(border=True):

                                    categorical_columns = df.select_dtypes(include=['object']).columns
                                    if len(categorical_columns) == 0:
                                        st.warning("There are no categorical variables in the dataset.Proceed with the original DataFrame")
                                        df = df.copy()
                                    else:
                                        for feature in df.columns: 
                                            if df[feature].dtype == 'object': 
                                                print('\n')
                                                print('feature:',feature)
                                                print(pd.Categorical(df[feature].unique()))
                                                print(pd.Categorical(df[feature].unique()).codes)
                                                df[feature] = pd.Categorical(df[feature]).codes
                                        st.success("**Categorical variables are encoded**")
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    
                                    st.markdown("##### 📋 Showing Top 3 rows of encoded data")
                                    st.dataframe(df.head(3),hide_index=True)
                                    st.download_button(label="📥 Download Encoded Data (for review) (.csv)", data=csv, file_name='encoded_data.csv', mime='text/csv')

                            with st.container(border=True):
                                
                                    if scaling_reqd == 'yes':     
                                        df = scale_features(df,scaling_method)
                                        st.success("**Data is scaled for further treatment**")
                                        csv = df.to_csv(index=False).encode('utf-8')
                                        
                                        st.markdown("##### 📋 Showing Top 3 rows of scaled data")
                                        st.dataframe(df.head(3),hide_index=True)
                                        st.download_button(label="📥 Download Scaled Data (for review) (.csv)", data=csv, file_name='scaled_data.csv', mime='text/csv')
                                    else:
                                        st.warning("Data is not scaled, orginal data is considered for further treatment | If Scalling required, change the options in the drop-down menu from **Hyperparameter** tab in the top.")

                            with st.expander("📌 Highly Correleated Features Summary", expanded=False):
                                    
                                    high_corr_pairs = (corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool)).stack().reset_index()
                                                       .rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: "Correlation"}))
                                    strong_corr = high_corr_pairs[high_corr_pairs["Correlation"].abs() > 0.5].sort_values(by="Correlation", ascending=False)
                                    if not strong_corr.empty:
                                        
                                        col1, col2 = st.columns((0.3,0.7))
                                        with col1:
                                            
                                            st.markdown("##### 🧠 Strongly Correlated Variable Pairs (|r| > 0.5)")
                                            st.table(strong_corr.reset_index(drop=True))

                                        with col2:
                                            
                                            st.markdown("##### 💬 Remarks and Suggestions")
                                            for _, row in strong_corr.iterrows():
                                                var1 = row["Feature 1"]
                                                var2 = row["Feature 2"]
                                                corr_val = row["Correlation"]
                                                strength = "strong" if abs(corr_val) > 0.75 else "moderate"
                                                direction = "positive" if corr_val > 0 else "negative"

                                                st.markdown(f"- **{var1}** and **{var2}** have a **{strength} {direction} correlation** of **{corr_val:.2f}**.")
                                                if abs(corr_val) > 0.75:
                                                    st.info(
                                                        f"🔁 These variables may carry redundant information. Consider dropping one of them "
                                                        f"or applying dimensionality reduction (e.g., PCA) to avoid multicollinearity."
                                                    )
                                    else:
                                        st.info("✅ No variable pairs found with correlation greater than 0.5.")  
                                        
                        with tabs[3]:
                            
                            with st.container(border=True):
                                
                                    #numeric_df = df.select_dtypes(include=['number'])
                                    pca = PCA(n_components=n_components)
                                    principal_components = pca.fit_transform(df)
                                    pc_columns = [f"PC{i+1}" for i in range(n_components)]
                                    pca_df = pd.DataFrame(data=principal_components, columns=pc_columns)
                                    st.markdown("##### 📉 Principal Components Analysis (PCA) | Reduced DataFrame")
                                    st.dataframe(pca_df,hide_index=True)
                                    
                            with st.container(border=True):
                                
                                col1,col2 = st.columns(2)
                                with col1:
                                    
                                    if n_components >= 3:

                                        fig = plt.figure(figsize=(10,8))
                                        ax = fig.add_subplot(111, projection='3d')
                                        ax.scatter(pca_df["PC1"], pca_df["PC2"], pca_df["PC3"],c='skyblue', edgecolor='k', s=40, marker="o",alpha=0.7)
                                        ax.set_title("3D PCA Projection", fontsize=9)
                                        ax.set_xlabel("PC1", fontsize=6)
                                        ax.set_ylabel("PC2", fontsize=6)
                                        ax.set_zlabel("PC3", fontsize=6)
                                        ax.tick_params(labelsize=6)
                                        st.pyplot(fig, use_container_width=True)

                                with col2:  
                                                                      
                                    explained_var = pca.explained_variance_ratio_
                                    st.markdown("##### 📊 Explained Variance by PCA Components")
                                    exp_var_df = pd.DataFrame({"Principal Component": pc_columns,"Explained Variance Ratio": explained_var})
                                    st.dataframe(exp_var_df,hide_index=True)

                        with tabs[4]:
                            
                            clustering_algorithms = {
                                        "KMeans": KMeans(n_clusters=n_clusters_final),
                                        "AffinityPropagation": AffinityPropagation(),
                                        "MeanShift": MeanShift(),
                                        "SpectralClustering": SpectralClustering(n_clusters=n_clusters_final),
                                        "AgglomerativeClustering": AgglomerativeClustering(n_clusters=n_clusters_final),
                                        "DBSCAN": DBSCAN(),
                                        "OPTICS": OPTICS(),
                                        "Birch": Birch(n_clusters=n_clusters_final),
                                        # Commented KModes for numeric PCA input only
                                        # "KModes": KModes(n_clusters=n_clusters_final, init='Cao', n_init=5, verbose=1)
                                    }
                                            
                            with st.spinner("Setting up and comparing models..."):

                                col1, col2 = st.columns(2)
                                with col1:   
                                    
                                        with st.container(border=True):
                                            st.markdown("##### 📈 Elbow Plot (KMeans WCSS vs Clusters)")
                                            k_range = range(2, 11)
                                            distortions = []
                                            fit_times = []
                                            for k in k_range:
                                                start = time.time()
                                                kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
                                                kmeans.fit(pca_df)
                                                distortions.append(kmeans.inertia_)
                                                fit_times.append(time.time() - start)
                                            knee = KneeLocator(k_range, distortions, curve='convex', direction='decreasing')
                                            elbow_k = knee.elbow
                                            fig, ax1 = plt.subplots(figsize=(7,4))
                                            color1 = 'tab:blue'
                                            color2 = 'mediumseagreen'
                                            ax1.plot(k_range, distortions, marker='o', linestyle='-', color=color1)
                                            ax1.set_xlabel("k", fontsize=9)
                                            ax1.set_ylabel("Distortion Score", color=color1, fontsize=9)
                                            ax1.tick_params(axis='y', labelcolor=color1)
                                            ax1.axvline(elbow_k, linestyle='--', color='black')
                                            score_at_elbow = distortions[k_range.index(elbow_k)]
                                            ax1.annotate(f'elbow at $k={elbow_k}$, $score={score_at_elbow:.0f}$',
                                                        xy=(elbow_k, score_at_elbow), xytext=(elbow_k+0.5, score_at_elbow + 500),
                                                        arrowprops=dict(arrowstyle='->', lw=1), fontsize=8)
                                            ax2 = ax1.twinx()
                                            ax2.plot(k_range, fit_times, marker='o', linestyle='--', color=color2)
                                            ax2.set_ylabel("fit time (seconds)", color=color2)
                                            ax2.tick_params(axis='y', labelcolor=color2)
                                            fig.tight_layout()
                                            st.pyplot(fig, use_container_width=True)             

                                with col2: 
                                     
                                        with st.container(border=True):
                                            st.markdown("##### 🌿 Dendrogram (Hierarchical Clustering)")
                                            Z = linkage(pca_df, method='ward')
                                            fig, ax = plt.subplots(figsize=(7,4))
                                            dendrogram(Z, ax=ax, truncate_mode='level',  no_labels=True, leaf_rotation=90, leaf_font_size=8, color_threshold=0.7 * max(Z[:, 2]))
                                            #ax.set_title("Dendrogram", fontsize=9)
                                            ax.set_xlabel("Samples", fontsize=9)
                                            ax.set_ylabel("Distance", fontsize=9)
                                            st.pyplot(fig, use_container_width=True)          

                            results = []
                            plots = []
                            for name, algo in clustering_algorithms.items():
                                try:
                                    labels = algo.fit_predict(pca_df)
                                    if len(set(labels)) <= 1:
                                        continue
                                    sil = silhouette_score(pca_df, labels)
                                    cal = calinski_harabasz_score(pca_df, labels)
                                    dav = davies_bouldin_score(pca_df, labels)
                                    results.append({
                                        "Algorithm": name,
                                        "Silhouette": round(sil, 3),
                                        "Calinski-Harabasz": round(cal, 3),
                                        "Davies-Bouldin": round(dav, 3),
                                        "Clusters": len(set(labels))
                                    })
                                    fig, ax = plt.subplots(figsize=(4,2.5))
                                    scatter = ax.scatter(pca_df["PC1"], pca_df["PC2"], c=labels, cmap='tab10', s=40, alpha=0.7)
                                    ax.set_title(f"{name}", fontsize=9)
                                    ax.set_xlabel("PC1", fontsize=7)
                                    ax.set_ylabel("PC2", fontsize=7)
                                    ax.tick_params(labelsize=6)
                                    plots.append((labels, name))
                                    pca_df[f"Cluster_{name}"] = labels
                                except Exception as e:
                                    st.warning(f"{name} failed: {e}")      

                        with tabs[5]:
                            
                            with st.container(border=True):
                                
                                    if results:
                                        st.markdown("##### 📋 Clustering Evaluation Summary")
                                        st.dataframe(pd.DataFrame(results), use_container_width=True)   

                        with tabs[6]:
                            
                            with st.container(border=True):

                                if plots:

                                    rows = [plots[i:i+4] for i in range(0, len(plots),4)]
                                    
                                    for row in rows:
                                        cols = st.columns(len(row)) 
                                        for col, (labels, name) in zip(cols, row):
                                            with col:
                                                fig = plt.figure(figsize=(5,4))
                                                if "PC3" in pca_df.columns:
                                                    ax = fig.add_subplot(111, projection='3d')
                                                    ax.scatter(pca_df["PC1"], pca_df["PC2"], pca_df["PC3"],c=labels, cmap='tab10', s=40, alpha=0.8, edgecolor='k')
                                                    ax.set_xlabel("PC1", fontsize=8)
                                                    ax.set_ylabel("PC2", fontsize=8)
                                                    ax.set_zlabel("PC3", fontsize=8)
                                                else:
                                                    ax = fig.add_subplot(111)
                                                    ax.scatter(pca_df["PC1"], pca_df["PC2"],c=labels, cmap='tab10', s=40, alpha=0.8, edgecolor='k')
                                                    ax.set_xlabel("PC1", fontsize=8)
                                                    ax.set_ylabel("PC2", fontsize=8)
                                                #ax.set_title(f"{name} Clustering", fontsize=9)
                                                ax.tick_params(labelsize=7)
                                                fig.tight_layout()
                                                st.markdown(f"**{name}**")
                                                st.pyplot(fig, use_container_width=True)
                                                
                                    st.markdown('--')
                                                                                                                                                                       
                        with tabs[7]:
                                                                    
                            with st.container(border=True):
                                
                                    try:
                                        kmeans_final = KMeans(n_clusters=n_clusters_final, random_state=42)
                                        df["Final_Cluster"] = kmeans_final.fit_predict(pca_df)
                                        df1["Final_Cluster"] = df["Final_Cluster"].values
                                        st.success(f"**✅ Final cluster label (KMeans with {n_clusters_final} clusters) added to original dataset as `Final_Cluster`**")
                                        st.dataframe(df[["Final_Cluster"] + df.columns.drop("Final_Cluster").tolist()][:5], use_container_width=True)
                                    except Exception as e:
                                        st.error(f"❌ Could not assign clusters to original data: {e}")

                            with st.container(border=True):
                                                                        
                                    st.markdown("##### 📊 Variable Behavior by Cluster")
                                    grouped_means = df1.groupby("Final_Cluster").mean(numeric_only=True).T

                                    variables = grouped_means.index.tolist()
                                    rows = [variables[i:i+3] for i in range(0, len(variables),3)]
                                    for row_vars in rows:
                                        row_cols = st.columns(len(row_vars))
                                        for var, col in zip(row_vars, row_cols):
                                            with col:
                                                fig, ax = plt.subplots(figsize=(4,2.5))
                                                grouped_means.loc[var].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                                                ax.set_title(var, fontsize=9)
                                                ax.set_xlabel("Cluster", fontsize=8)
                                                ax.set_ylabel("Values", fontsize=8)
                                                ax.tick_params(axis='x', rotation=0, labelsize=8)
                                                ax.tick_params(axis='y', labelsize=8)
                                                st.pyplot(fig, use_container_width=True)
