import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, predict_model, pull, plot_model, create_model, ensemble_model, blend_models, stack_models, tune_model, save_model

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
    "KS Statistic Plot":  'ks'
}

def load_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xls') or file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None

def classification_pycaret():
    st.title("AutoML - Classification using PyCaret")
    
    uploaded_file = st.file_uploader("**Choose a file**", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        df = load_file(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())

        st.header("Dataset Configuration")
        target = st.selectbox("Choose the target column:", df.columns)
        
        if target:
            feature_list = df.columns.tolist()
            feature_list.remove(target)
            features = st.multiselect("Select features to include:", feature_list)
            if features:
                train_size = st.slider("Set the training data size:", 0.1, 0.9, 0.8)
                validation_size = st.slider("Set the validation data size:", 0.1, 0.9 - train_size, 0.1)
                test_size = 1 - train_size - validation_size
                features.append(target)
                data = df[features].sample(frac=train_size, random_state=786).reset_index(drop=True)
                data_unseen = df[features].drop(data.index).reset_index(drop=True)
                
                if st.button("Submit"):
                    with st.spinner("Setting up and comparing models..."):
                        s = setup(data, target=target, session_id=123)
                        st.markdown('<p style="color:#4FFF33">Setup Successfully Completed!</p>', unsafe_allow_html=True)
                        st.dataframe(pull())
                        
                        best_model = compare_models()
                        results = pull()
                        st.write("### Best Model: ", results['Model'].iloc[0])
                        st.write('#### Comparing All Models')
                        st.dataframe(results)
                        
                        model_option = st.selectbox("Select a model option", ["Best Model", "Specific Model", "Ensemble Model", "Blending", "Stacking"])
                        if model_option == "Specific Model":
                            model_name = st.selectbox("Choose the model name", results['Model'].to_list())
                        elif model_option in ["Ensemble Model", "Blending", "Stacking"]:
                            model_name = st.selectbox("Choose the model", results['Model'].to_list())
                            method = st.selectbox("Choose the method: ", ['Bagging', 'Boosting'] if model_option == "Ensemble Model" else ['soft', 'hard'])
                        
                        tune_model_option = st.checkbox("Tune the model")
                        selected_metrics = st.multiselect("Select classification metrics to evaluate", options=list(metrics_dict.keys()))
                        uploaded_file_test = st.file_uploader("Upload CSV or Excel test file (optional)", type=["csv", "xlsx"], key='test')

                        if st.button("Run AutoML"):
                            with st.spinner("Running AutoML..."):
                                if model_option == "Best Model":
                                    model_cls = best_model
                                    model_name = results['Model'].iloc[0]
                                elif model_option == "Specific Model":
                                    model_cls = create_model(results[results['Model'] == model_name].index[0])
                                elif model_option == "Ensemble Model":
                                    model_cls = ensemble_model(create_model(results[results['Model'] == model_name].index[0]), method=method)
                                elif model_option == "Blending":
                                    blend_models_list = st.multiselect("Choose the models for blending", results['Model'].to_list())
                                    model_cls = blend_models(estimator_list=[create_model(results[results['Model'] == m].index[0]) for m in blend_models_list], method=method)
                                elif model_option == "Stacking":
                                    stack_models_list = st.multiselect("Choose models for stacking", results['Model'].to_list())
                                    model_cls = stack_models([create_model(results[results['Model'] == m].index[0]) for m in stack_models_list])
                                
                                if tune_model_option:
                                    final_model = tune_model(model_cls)
                                else:
                                    final_model = model_cls

                                if selected_metrics:
                                    for metric in selected_metrics:
                                        plot_model(final_model, plot=metrics_dict[metric], display_format='streamlit')

                                pred_holdout = predict_model(final_model)
                                st.write('#### Predictions from holdout set (validation set)')
                                st.dataframe(pred_holdout)
                                
                                if uploaded_file_test:
                                    test_data = load_file(uploaded_file_test)
                                    if target in test_data.columns:
                                        test_data = test_data.drop(target, axis=1)
                                    test_pred = predict_model(final_model, data=test_data)
                                    st.write("### Prediction on Test Data")
                                    st.dataframe(test_pred)

                                st.success("AutoML process completed!")
                                
classification_pycaret()
