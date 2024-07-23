import streamlit as st

# Define a list of projects with their names and URLs
projects = [
    {"name": "Project Alpha", "url": "https://example.com/project-alpha"},
    {"name": "Project Beta", "url": "https://example.com/project-beta"},
    {"name": "Project Gamma", "url": "https://example.com/project-gamma"},
]

# Streamlit App Title centered using HTML and CSS
st.markdown(
    """
    <h1 style='text-align: center; color: rainbow;'>Digital eWatch | BU Care Chemicals | Energy Consumption | Regression & Anomaly Analysis | v3.0</h1>
    """,
    unsafe_allow_html=True
)

st.write("Click on a project name to navigate to the specific webpage:")

# Display the project names as clickable links
for project in projects:
    st.markdown(f"[{project['name']}]({project['url']})")

# Additional information about developer, preparer, and forecasting app button on the same line
st.markdown(
    """
    <div style='text-align: center;'>
        Developed by: <strong style='color: blue;'>E&PT - Digital Solutions</strong> | prepared by: 
        <a href="mailto:avijit.chakraborty@clariant.com">Avijit Chakraborty</a> | 
        <a href="http://10.72.97.157:8519/" target="_blank">
            <button style="padding:10px 20px; font-size:16px; background-color:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer;">
                Forecasting App
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# Optional: Sidebar for additional actions or navigation
st.sidebar.title("Navigation")
for project in projects:
    st.sidebar.markdown(f"[{project['name']}]({project['url']})")

st.sidebar.markdown(
    """
    <a href="http://10.72.97.157:8519/" target="_blank">
        <button style="padding:10px 20px; font-size:16px; background-color:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer;">
            Forecasting App
        </button>
    </a>
    """,
    unsafe_allow_html=True
)
