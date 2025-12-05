import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title('Car Price Prediction')

@st.cache_resource
def train_model():
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
    from sklearn.linear_model import Ridge
    
    def clean_numeric(df):
        df = df.copy()
        for col in ['mileage', 'engine', 'max_power']:
            df[col] = (df[col]
                      .astype(str)
                      .str.replace(r'[^\d\.]+', '', regex=True)
                      .replace('', np.nan)
                      .astype(float))
        return df
    
    def drop_torque(df):
        return df.drop(columns=['torque'], errors='ignore')
    
    def add_brand(df):
        df = df.copy()
        df['brand'] = df['name'].str.split().str[0]
        df = df.drop(columns=['name'])
        return df
    
    cleaner_numeric = FunctionTransformer(clean_numeric)
    drop_torque_tf = FunctionTransformer(drop_torque)
    brand_tf = FunctionTransformer(add_brand)
    
    numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    categorical_features = ['fuel', 'seller_type', 'seats', 'transmission', 'owner', 'brand']
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ], remainder='drop')
    
    pipeline = Pipeline([
        ('drop_torque', drop_torque_tf),
        ('cleaner_numeric', cleaner_numeric),
        ('brand_creator', brand_tf),
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=1.0))
    ])
    
    url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
    df = pd.read_csv(url)
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']
    
    pipeline.fit(X, y)
    return pipeline

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
    return pd.read_csv(url)

pipeline = train_model()
df_raw = load_data()

with st.sidebar:
    st.header("Input Car Details")
    
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", 1980, 2024, 2015)
        km_driven = st.number_input("KM Driven", 0, 1000000, 50000)
        mileage = st.number_input("Mileage", 0.0, 50.0, 20.0)
    with col2:
        engine = st.number_input("Engine (CC)", 0, 5000, 1200)
        max_power = st.number_input("Max Power", 0.0, 500.0, 80.0)
        seats = st.selectbox("Seats", sorted(df_raw['seats'].dropna().unique().astype(int)))
    
    fuel = st.selectbox("Fuel Type", df_raw['fuel'].unique())
    seller_type = st.selectbox("Seller Type", df_raw['seller_type'].unique())
    transmission = st.selectbox("Transmission", df_raw['transmission'].unique())
    owner = st.selectbox("Owner", df_raw['owner'].unique())
    
    brand_names = df_raw['name'].str.split().str[0].unique()
    brand = st.selectbox("Brand", sorted(brand_names))
    
    if st.button("Predict Price", type="primary"):
        input_data = pd.DataFrame({
            'name': [f"{brand} Model"],
            'year': [year],
            'km_driven': [km_driven],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'seats': [seats],
            'torque': ['100Nm@2000rpm']
        })
        
        prediction = pipeline.predict(input_data)[0]
        st.session_state.prediction = prediction
        st.session_state.input_data = input_data.iloc[0].to_dict()

if 'prediction' in st.session_state:
    st.success(f"### Predicted Price: ₹{st.session_state.prediction:,.2f}")
    
    st.subheader("Input Summary")
    cols = st.columns(3)
    input_data = st.session_state.input_data
    
    with cols[0]:
        st.write(f"**Year:** {input_data['year']}")
        st.write(f"**KM Driven:** {input_data['km_driven']:,}")
        st.write(f"**Mileage:** {input_data['mileage']}")
    with cols[1]:
        st.write(f"**Engine:** {input_data['engine']} CC")
        st.write(f"**Max Power:** {input_data['max_power']} bhp")
        st.write(f"**Seats:** {input_data['seats']}")
    with cols[2]:
        st.write(f"**Fuel:** {input_data['fuel']}")
        st.write(f"**Seller Type:** {input_data['seller_type']}")
        st.write(f"**Transmission:** {input_data['transmission']}")
        st.write(f"**Owner:** {input_data['owner']}")
        st.write(f"**Brand:** {brand}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Data")
    st.dataframe(df_raw.head(10), use_container_width=True)

with col2:
    st.subheader("Data Statistics")
    st.write(f"**Total rows:** {len(df_raw):,}")
    st.write(f"**Price range:** ₹{df_raw['selling_price'].min():,.0f} - ₹{df_raw['selling_price'].max():,.0f}")
    st.write(f"**Average price:** ₹{df_raw['selling_price'].mean():,.0f}")
    st.write(f"**Year range:** {df_raw['year'].min()} - {df_raw['year'].max()}")

st.subheader("Price Distribution")
st.bar_chart(df_raw['selling_price'].value_counts().head(20))
