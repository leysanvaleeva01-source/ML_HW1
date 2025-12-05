

st.title('EDA')
st.write("Cмотрим графики")


#Пришлось заново строить предобработку, потому что streamlit c ней не подружился
def preprocess_data(df, reference_df=None):
    """Вся предобработка, которую мы использовали"""
    df = df.copy()
    columns_to_clean = ['mileage', 'engine', 'max_power']
    for column in columns_to_clean:
        df[column] = df[column].astype(str).str.replace(r'[^\d\.]+', '', regex=True)
        df[column] = pd.to_numeric(df[column], errors='coerce').astype('float64')
    if 'torque' in df.columns:
        df = df.drop('torque', axis=1)
    if 'name' in df.columns:
        df['brand'] = df['name'].str.split().str[0]
        df = df.drop('name', axis=1)
        col = "brand"
        if reference_df is not None:
            train_cats = set(reference_df[col].unique())
            test_cats = set(df[col].unique())
            unknown = test_cats - train_cats
            if unknown:
                most_frequent = reference_df[col].mode()[0]
                df[col] = df[col].replace(list(unknown), most_frequent)
    
    return df


@st.cache_resource
def load_pipeline():
    """Загружает обученный пайплайн"""
    try:
        # Проверяем, существует ли файл локально
        if os.path.exists('final_pipeline.pkl'):
            with open('final_pipeline.pkl', 'rb') as f:
                pipeline = pickle.load(f)
            st.success("Модель загружена успешно!")
            return pipeline
        else:
            # Если файла нет локально, пробуем загрузить из текущей директории
            st.warning("Файл final_pipeline.pkl не найден в текущей директории")
            st.info("Текущая рабочая директория: " + os.getcwd())
            st.info("Содержимое директории: " + str(os.listdir('.')))
            return None
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None
@st.cache_data
def load_and_preprocess_data():
    try:
        # Загружаем исходные данные
        train_url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
        test_url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv"
        
        df_train_raw = pd.read_csv(train_url)
        df_test_raw = pd.read_csv(test_url)
        
        y_train = df_train_raw['selling_price']
        y_test = df_test_raw['selling_price']
        X_train_raw = df_train_raw.drop('selling_price', axis=1)
        X_test_raw = df_test_raw.drop('selling_price', axis=1)
        
        return X_train_raw, y_train, X_test_raw, y_test
        
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None, None, None, None

pipeline = load_pipeline()
X_train_raw, y_train, X_test_raw, y_test = load_and_preprocess_data()
