import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import sys
import platform

st.set_page_config(page_title="Library Versions", layout="wide")
st.title("📦 Library Versions Check")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Core Python & Data Science")
    st.write(f"**Python:** {sys.version.split()[0]}")
    st.write(f"**Pandas:** {pd.__version__}")
    st.write(f"**NumPy:** {np.__version__}")
    st.write(f"**Scikit-learn:** {sklearn.__version__}")
    
with col2:
    st.subheader("Visualization")
    try:
        import matplotlib.pyplot as plt
        st.write(f"**Matplotlib:** {plt.__version__}")
    except:
        st.write("**Matplotlib:** Not available")
    
    try:
        import seaborn as sns
        st.write(f"**Seaborn:** {sns.__version__}")
    except:
        st.write("**Seaborn:** Not available")

st.subheader("Sklearn Components")
try:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
    from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import r2_score
    
    sklearn_components = [
        "Pipeline", "ColumnTransformer", "SimpleImputer",
        "FunctionTransformer", "StandardScaler", "OneHotEncoder",
        "LinearRegression", "Lasso", "Ridge", "ElasticNet",
        "train_test_split", "GridSearchCV", "r2_score"
    ]
    
    status_data = []
    for comp in sklearn_components:
        try:
            exec(f"test = {comp}")
            status_data.append({"Component": comp, "Status": "✅ Available"})
        except:
            status_data.append({"Component": comp, "Status": "⚠️ Not available"})
    
    st.dataframe(pd.DataFrame(status_data))
    
except Exception as e:
    st.error(f"Error checking sklearn: {str(e)[:100]}")

st.subheader("System Information")
sys_info = {
    "Platform": platform.platform(),
    "Python Version": sys.version,
    "System Encoding": sys.getdefaultencoding()
}

for key, value in sys_info.items():
    st.write(f"**{key}:** {value}")

if st.button("Check All Packages"):
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        st.text_area("All Installed Packages", result.stdout, height=400)
    except:
        st.error("Could not get package list")
