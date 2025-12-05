import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import sys
import platform

st.set_page_config(page_title="System Info", layout="wide")
st.title('System Information')

col1, col2 = st.columns(2)

with col1:
    st.subheader("Python Information")
    st.write(f"**Python Version:** {sys.version}")
    st.write(f"**Platform:** {platform.platform()}")
    st.write(f"**Executable:** {sys.executable}")

with col2:
    st.subheader("Core Libraries")
    
    lib_versions = {
        "Streamlit": ("streamlit", "__version__"),
        "Pandas": ("pandas", "__version__"),
        "NumPy": ("numpy", "__version__"),
        "Scikit-learn": ("sklearn", "__version__"),
        "SciPy": ("scipy", "__version__")
    }
    
    for lib_name, (module_name, attr) in lib_versions.items():
        try:
            module = __import__(module_name)
            version = getattr(module, attr, "N/A")
            st.write(f"**{lib_name}:** {version}")
        except ImportError:
            st.write(f"**{lib_name}:** Not installed")

st.subheader("Installed Packages")
try:
    import pkg_resources
    
    packages = []
    for dist in pkg_resources.working_set:
        packages.append(f"{dist.project_name}=={dist.version}")
    
    packages.sort()
    
    st.text_area("All installed packages:", "\n".join(packages), height=300)
except:
    st.warning("Could not get full package list")

st.subheader("Pip Freeze Output")
try:
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                          capture_output=True, text=True)
    st.text_area("pip freeze output:", result.stdout, height=300)
except:
    st.error("Could not run pip freeze")
