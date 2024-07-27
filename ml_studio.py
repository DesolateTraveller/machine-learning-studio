import streamlit as st
import pandas as pd
import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class
from ydata_profiling import ProfileReport
import os

def load_file(file):
    if file.name.endswith('csv'):
        return pd.read_csv(file)
    elif file.name.endswith('xls') or file.name.endswith('xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None

def generate_sweetviz_report(df):
    report = sv.analyze(df)
    report.show_html(filepath='sweetviz_report.html', open_browser=False)
    return 'sweetviz_report.html'

def generate_autoviz_report(df, filename):
    AV = AutoViz_Class()
    report = AV.AutoViz(filename, dfte=df)
    return report

def generate_ydata_profiling_report(df):
    profile = ProfileReport(df, explorative=True)
    profile.to_file("ydata_profiling_report.html")
    return 'ydata_profiling_report.html'

# Main App
st.title("Exploratory Data Analysis (EDA) App")

file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx"], accept_multiple_files=False)

if file is not None:
    df = load_file(file)
    st.write("Preview of Data")
    st.dataframe(df.head())

    eda_option = st.selectbox(
        "Choose EDA Tool",
        ["Sweetviz", "AutoViz", "ydata Profiling"]
    )

    if st.button("Generate EDA Report"):
        if eda_option == "Sweetviz":
            report_file = generate_sweetviz_report(df)
            st.success("Sweetviz report generated!")
            with open(report_file, 'rb') as f:
                st.download_button('Download Sweetviz Report', f, file_name='sweetviz_report.html')

        elif eda_option == "AutoViz":
            report_file = 'uploaded_file.csv'
            df.to_csv(report_file, index=False)
            st.success("AutoViz report generated!")
            generate_autoviz_report(df, report_file)
            with open(report_file, 'rb') as f:
                st.download_button('Download AutoViz Report', f, file_name='autoviz_report.html')

        elif eda_option == "ydata Profiling":
            report_file = generate_ydata_profiling_report(df)
            st.success("ydata Profiling report generated!")
            with open(report_file, 'rb') as f:
                st.download_button('Download ydata Profiling Report', f, file_name='ydata_profiling_report.html')

else:
    st.info("Please upload a file to start the EDA process.")
