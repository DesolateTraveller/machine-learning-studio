import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_file(file):
    if file.name.endswith('csv'):
        return pd.read_csv(file)
    elif file.name.endswith('xls') or file.name.endswith('xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None

def label_encode(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df

def onehot_encode(df, column):
    ohe = OneHotEncoder(sparse=False)
    encoded_cols = ohe.fit_transform(df[[column]])
    encoded_df = pd.DataFrame(encoded_cols, columns=[f"{column}_{cat}" for cat in ohe.categories_[0]])
    df = df.drop(column, axis=1).join(encoded_df)
    return df

# Main App
st.title("Feature Encoding App")

file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx"], accept_multiple_files=False)

if file is not None:
    df = load_file(file)
    st.write("Preview of Data")
    st.dataframe(df.head())

    target_variable = st.selectbox("Choose Target Variable", options=["None"] + list(df.columns), key="target_variable")

    if target_variable == "None":
        st.warning("Please choose a target variable to proceed with the encoding.")
    else:
        # Display the dataframe's column names
        st.write("Dataframe Columns")
        st.write(df.columns.tolist())

        # Select columns to encode
        columns_to_encode = st.multiselect("Select columns to encode", options=df.select_dtypes(include=['object']).columns)

        if len(columns_to_encode) > 0:
            encoding_method = st.selectbox("Choose Encoding Method", ["Label Encoding", "One-Hot Encoding"])

            if st.button("Encode"):
                if encoding_method == "Label Encoding":
                    for col in columns_to_encode:
                        df = label_encode(df, col)
                elif encoding_method == "One-Hot Encoding":
                    for col in columns_to_encode:
                        df = onehot_encode(df, col)
                st.write("Encoded Data")
                st.dataframe(df.head())
        else:
            st.info("Please select at least one categorical column to encode.")

        # Provide an option to download the encoded dataset
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Encoded Data", data=csv, file_name='encoded_data.csv', mime='text/csv')
else:
    st.info("Please upload a file to start the encoding process.")
