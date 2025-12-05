import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title('Car Price Prediction')

@st.cache_resource
def load_model():
    with open("simple_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_preprocessed_data():
    with open("preprocessed_data.pkl", "rb") as f:
        return pickle.load(f)

def clean_numeric(df):
    df = df.copy()
    for col in ['mileage', 'engine', 'max_power']:
        if col in df.columns:
            df[col] = (df[col]
                      .astype(str)
                      .str.replace(r'[^\d\.]+', '', regex=True)
                      .replace('', np.nan)
                      .astype(float))
    return df

def preprocess_input(df):
    df = df.copy()
    df = df.drop(columns=['torque'], errors='ignore')
    df = clean_numeric(df)
    df['brand'] = df['name'].str.split().str[0]
    df = df.drop(columns=['name'], errors='ignore')
    return df

try:
    model = load_model()
    data = load_preprocessed_data()
    
    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch CSV Prediction", "EDA & Statistics"])
    
    with tab1:
        st.header("Single Car Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Year", 1980, 2024, 2015)
            km_driven = st.number_input("KM Driven", 0, 1000000, 50000)
            mileage = st.number_input("Mileage", 0.0, 50.0, 20.0)
            engine = st.number_input("Engine (CC)", 0, 5000, 1200)
            max_power = st.number_input("Max Power", 0.0, 500.0, 80.0)
        
        with col2:
            # Используем данные из предобработанного датасета
            fuel_options = data['X_train']['fuel'].unique() if 'fuel' in data['X_train'].columns else []
            seller_options = data['X_train']['seller_type'].unique() if 'seller_type' in data['X_train'].columns else []
            transmission_options = data['X_train']['transmission'].unique() if 'transmission' in data['X_train'].columns else []
            owner_options = data['X_train']['owner'].unique() if 'owner' in data['X_train'].columns else []
            seats_options = sorted(data['X_train']['seats'].dropna().unique().astype(int)) if 'seats' in data['X_train'].columns else []
            brand_options = sorted(data['X_train']['brand'].unique()) if 'brand' in data['X_train'].columns else []
            
            fuel = st.selectbox("Fuel Type", fuel_options) if len(fuel_options) > 0 else st.selectbox("Fuel Type", ["Petrol", "Diesel"])
            seller_type = st.selectbox("Seller Type", seller_options) if len(seller_options) > 0 else st.selectbox("Seller Type", ["Individual", "Dealer"])
            transmission = st.selectbox("Transmission", transmission_options) if len(transmission_options) > 0 else st.selectbox("Transmission", ["Manual", "Automatic"])
            owner = st.selectbox("Owner", owner_options) if len(owner_options) > 0 else st.selectbox("Owner", ["First Owner", "Second Owner"])
            seats = st.selectbox("Seats", seats_options) if len(seats_options) > 0 else st.selectbox("Seats", [5, 7, 8])
            brand = st.selectbox("Brand", brand_options) if len(brand_options) > 0 else st.selectbox("Brand", ["Maruti", "Hyundai", "Honda"])
        
        if st.button("Predict Price", type="primary"):
            input_dict = {
                'name': f"{brand} Model",
                'year': year,
                'km_driven': km_driven,
                'fuel': fuel,
                'seller_type': seller_type,
                'transmission': transmission,
                'owner': owner,
                'mileage': mileage,
                'engine': engine,
                'max_power': max_power,
                'seats': seats,
                'torque': '100Nm@2000rpm'
            }
            
            input_df = pd.DataFrame([input_dict])
            processed_df = preprocess_input(input_df)
            
            for col in data['feature_names']:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            
            processed_df = processed_df[data['feature_names']]
            prediction = model.predict(processed_df)[0]
            
            st.success(f"### Predicted Price: ₹{prediction:,.2f}")
            
            st.subheader("Processed Features")
            st.dataframe(processed_df, use_container_width=True)
    
    with tab2:
        st.header("Batch CSV Prediction")
        
        uploaded_file = st.file_uploader("Upload CSV with car data", type="csv")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            
            required_cols = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 
                            'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
            
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                st.success(f"Loaded {len(batch_df)} cars")
                
                if st.button("Process and Predict", type="primary"):
                    with st.spinner("Processing..."):
                        original_df = batch_df.copy()
                        processed_batch = preprocess_input(batch_df)
                        
                        for col in data['feature_names']:
                            if col not in processed_batch.columns:
                                processed_batch[col] = 0
                        
                        processed_batch = processed_batch[data['feature_names']]
                        predictions = model.predict(processed_batch)
                        
                        original_df['predicted_price'] = predictions
                        
                        st.subheader("Predictions")
                        st.dataframe(original_df[['name', 'year', 'fuel', 'engine', 'predicted_price']], 
                                   use_container_width=True)
                        
                        csv = original_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
    
    with tab3:
        st.header("EDA & Statistics")
        
        if 'X_train' in data and 'y_train' in data:
            X_train = data['X_train']
            y_train = data['y_train']
            
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(X_train))
            with col2:
                st.metric("Features", len(X_train.columns))
            with col3:
                st.metric("Avg Price", f"₹{y_train.mean():,.0f}")
            with col4:
                st.metric("Price Std", f"₹{y_train.std():,.0f}")
            
            st.subheader("Price Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(y_train, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Price (₹)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Car Prices')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("Numerical Features vs Price")
            num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
            num_cols = [col for col in num_cols if col in X_train.columns]
            
            selected_num = st.selectbox("Select numerical feature", num_cols)
            
            if selected_num:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(X_train[selected_num], y_train, alpha=0.5, s=10)
                ax.set_xlabel(selected_num)
                ax.set_ylabel('Price (₹)')
                ax.set_title(f'{selected_num} vs Price')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            st.subheader("Categorical Features Analysis")
            cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'seats']
            cat_cols = [col for col in cat_cols if col in X_train.columns]
            
            selected_cat = st.selectbox("Select categorical feature", cat_cols)
            
            if selected_cat:
                fig, ax = plt.subplots(figsize=(12, 6))
                cat_stats = pd.DataFrame({
                    'category': X_train[selected_cat],
                    'price': y_train
                })
                cat_means = cat_stats.groupby('category')['price'].mean().sort_values(ascending=False)
                
                if len(cat_means) > 20:
                    cat_means = cat_means.head(20)
                
                bars = ax.bar(range(len(cat_means)), cat_means.values)
                ax.set_xticks(range(len(cat_means)))
                ax.set_xticklabels(cat_means.index, rotation=45, ha='right')
                ax.set_xlabel(selected_cat)
                ax.set_ylabel('Average Price (₹)')
                ax.set_title(f'Average Price by {selected_cat}')
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar, price in zip(bars, cat_means.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'₹{price:,.0f}', ha='center', va='bottom', fontsize=8)
                
                st.pyplot(fig)
            
            st.subheader("Correlation Heatmap")
            numeric_data = X_train[num_cols].copy()
            numeric_data['price'] = y_train.values
            
            if len(numeric_data.columns) > 1:
                corr = numeric_data.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, ax=ax)
                ax.set_title('Correlation Matrix')
                st.pyplot(fig)
            
            st.subheader("Feature Statistics")
            stats_df = X_train.describe().T
            stats_df['count'] = stats_df['count'].astype(int)
            st.dataframe(stats_df, use_container_width=True)
            
        else:
            st.warning("Preprocessed data not available for EDA")

except FileNotFoundError:
    st.error("Required files not found. Please upload:")
    st.write("- simple_model.pkl")
    st.write("- preprocessed_data.pkl")
    
    # Альтернатива: показать пример данных
    if st.button("Show Example EDA"):
        st.info("Example EDA would be shown here if data was available")

except Exception as e:
    st.error(f"Error loading data: {str(e)[:200]}")
