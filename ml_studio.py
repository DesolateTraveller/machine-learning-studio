import streamlit as st
import pandas as pd
import sweetviz as sv
from ydata_profiling import ProfileReport
import base64
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
    with open('sweetviz_report.html', 'r') as f:
        html = f.read()
    return html

def generate_ydata_profiling_report(df):
    profile = ProfileReport(df, explorative=True)
    profile.to_file("ydata_profiling_report.html")
    with open('ydata_profiling_report.html', 'r') as f:
        html = f.read()
    return html

def display_html_report(html, title):
    b64 = base64.b64encode(html.encode()).decode()
    iframe = f'<iframe src="data:text/html;base64,{b64}" width="100%" height="800px"></iframe>'
    st.markdown(f"### {title}")
    st.markdown(iframe, unsafe_allow_html=True)

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
            with st.spinner("Generating Sweetviz report..."):
                report_html = generate_sweetviz_report(df)
                with st.expander("Sweetviz Report"):
                    display_html_report(report_html, "Sweetviz Report")
                st.success("Sweetviz report generated!")


        elif eda_option == "ydata Profiling":
            with st.spinner("Generating ydata Profiling report..."):
                report_html = generate_ydata_profiling_report(df)
                with st.expander("ydata Profiling Report"):
                    display_html_report(report_html, "ydata Profiling Report")
                st.success("ydata Profiling report generated!")
else:
    st.info("Please upload a file to start the EDA process.")
