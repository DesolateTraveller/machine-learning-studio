import streamlit as st
import pandas as pd
import pygwalker as pyg

# Sample DataFrame for demonstration
# Replace this with your actual data
data = {
    "date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
    "value1": pd.Series(range(100)),
    "value2": pd.Series(range(100, 200))
}
df = pd.DataFrame(data)

# Streamlit app setup
st.set_page_config(page_title="Interactive Data Visualization with Pygwalker", layout="wide")

st.title("Interactive Data Visualization with Pygwalker")

st.sidebar.title("Configuration")
selected_columns = st.sidebar.multiselect("Select columns to visualize", options=df.columns.tolist(), default=df.columns.tolist())

# Filter the DataFrame based on the selected columns
if selected_columns:
    filtered_df = df[selected_columns]
else:
    filtered_df = df  # Fallback to full DataFrame if no columns are selected

st.sidebar.write("## Filtered DataFrame")
st.sidebar.write(filtered_df)

# Pygwalker visualization
st.write("## Pygwalker Visualization")
pyg.walk(filtered_df)

# Optional: Display the DataFrame
st.write("## DataFrame")
st.write(filtered_df)
