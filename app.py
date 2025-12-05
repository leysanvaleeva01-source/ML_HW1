import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pickle
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE

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
    st.write(f"**Matplotlib:** {plt.__version__}")
    st.write(f"**Seaborn:** {sns.__version__}")

st.subheader("All Versions Table")
versions_data = {
    "Library": [
        "Streamlit", "Pandas", "NumPy", "Scikit-learn",
        "Matplotlib", "Seaborn", "Pickle", "Random"
    ],
    "Version": [
        st.__version__, pd.__version__, np.__version__, sklearn.__version__,
        plt.__version__, sns.__version__, "Built-in", "Built-in"
    ]
}
st.dataframe(pd.DataFrame(versions_data))

st.subheader("Sklearn Components Check")
sklearn_modules = [
    "Pipeline", "ColumnTransformer", "SimpleImputer",
    "FunctionTransformer", "StandardScaler", "OneHotEncoder",
    "LinearRegression", "Lasso", "Ridge", "ElasticNet",
    "train_test_split", "GridSearchCV", "r2_score"
]

sklearn_status = []
for module in sklearn_modules:
    try:
        exec(f"from sklearn import {module}")
        sklearn_status.append({"Module": module, "Status": "✅ Available"})
    except ImportError:
        sklearn_status.append({"Module": module, "Status": "❌ Not Available"})

st.dataframe(pd.DataFrame(sklearn_status))

try:
    from ydata_profiling import ProfileReport
    st.success(f"✅ ydata-profiling: Available")
except ImportError:
    st.warning("❌ ydata-profiling: Not installed")

st.subheader("System Information")
import sys
import platform

sys_info = {
    "Platform": platform.platform(),
    "Python Version": sys.version,
    "Python Executable": sys.executable,
    "System Encoding": sys.getdefaultencoding()
}

for key, value in sys_info.items():
    st.write(f"**{key}:** {value}")

if st.button("Check All Packages"):
    try:
        import pkg_resources
        packages = []
        for dist in pkg_resources.working_set:
            packages.append(f"{dist.project_name}=={dist.version}")
        
        packages.sort()
        st.text_area("All Installed Packages", "\n".join(packages), height=400)
    except:
        st.error("Could not get package list")
