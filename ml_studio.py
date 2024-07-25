import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def load_file(file):
    if file.name.endswith('csv'):
        return pd.read_csv(file)
    elif file.name.endswith('xls') or file.name.endswith('xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None

def scale_features(df, columns, method):
    if method == 'Standard Scaling':
        scaler = StandardScaler()
    elif method == 'Min-Max Scaling':
        scaler = MinMaxScaler()
    elif method == 'Robust Scaling':
        scaler = RobustScaler()
    
    df[columns] = scaler.fit_transform(df[columns])
    return df

# Main App
st.title("Feature Scaling App")

file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx"], accept_multiple_files=False)

if file is not None:
    df = load_file(file)
    st.write("Preview of Data")
    st.dataframe(df.head())

    target_variable = st.selectbox("Choose Target Variable", options=["None"] + list(df.columns), key="target_variable")

    if target_variable == "None":
        st.warning("Please choose a target variable to proceed with the scaling.")
    else:
        # Identify numerical columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

        if len(numerical_columns) == 0:
            st.info("There are no numerical variables in the dataset.")
        else:
            st.subheader("Feature Scaling:")
            columns_to_scale = st.multiselect("Select columns to scale", options=numerical_columns)

            if len(columns_to_scale) > 0:
                scaling_method = st.selectbox("Choose Scaling Method", ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"])

                if st.button("Scale"):
                    df = scale_features(df, columns_to_scale, scaling_method)
                    st.write("Scaled Data")
                    st.dataframe(df.head())
            else:
                st.info("Please select at least one numerical column to scale.")

            # Provide an option to download the scaled dataset
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Scaled Data", data=csv, file_name='scaled_data.csv', mime='text/csv')
else:
    st.info("Please upload a file to start the scaling process.")
