import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import io

st.set_page_config(
    page_title="Предсказатель цен на автомобили",
    layout="wide"
)

st.title('Предсказатель цен на автомобили')
st.write("Предсказание стоимости автомобилей на основе характеристик")

@st.cache_resource
def load_pipeline():
    try:
        with open('final_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        st.error("Файл final_pipeline.pkl не найден")
        return None

@st.cache_data
def load_and_preprocess_data():
    """Загружает исходные данные и применяет предобработку"""
    try:
        # Загружаем исходные данные
        train_url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
        test_url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv"
        
        df_train_raw = pd.read_csv(train_url)
        df_test_raw = pd.read_csv(test_url)
        
        # Сохраняем целевые переменные
        y_train = df_train_raw['selling_price']
        y_test = df_test_raw['selling_price']
        
        # Удаляем целевые переменные для предобработки
        X_train_raw = df_train_raw.drop('selling_price', axis=1)
        X_test_raw = df_test_raw.drop('selling_price', axis=1)
        
        return X_train_raw, y_train, X_test_raw, y_test
        
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None, None, None, None

pipeline = load_pipeline()
X_train_raw, y_train, X_test_raw, y_test = load_and_preprocess_data()

# EDA раздел - применяем предобработку к данным
st.header('Анализ данных (EDA) после предобработки')

if pipeline is not None and X_train_raw is not None:
    # Применяем только preprocessing часть пайплайна
    with st.spinner("Применяем предобработку данных..."):
        try:
            # Если пайплайн имеет метод transform для preprocessing
            if hasattr(pipeline, 'named_steps'):
                # Создаем preprocessing pipeline (без модели)
                from sklearn.pipeline import Pipeline
                
                # Извлекаем все шаги кроме последнего (модель)
                preprocess_steps = []
                for step_name, step_transformer in pipeline.named_steps.items():
                    if step_name != 'model':
                        preprocess_steps.append((step_name, step_transformer))
                
                # Создаем preprocessing pipeline
                preprocess_pipeline = Pipeline(preprocess_steps)
                
                # Применяем предобработку
                X_train_processed = preprocess_pipeline.transform(X_train_raw)
                X_test_processed = preprocess_pipeline.transform(X_test_raw)
                
                # Создаем DataFrame с обработанными данными
                # Получаем имена признаков
                feature_names = []
                if hasattr(preprocess_pipeline, 'get_feature_names_out'):
                    feature_names = list(preprocess_pipeline.get_feature_names_out())
                else:
                    # Если не можем получить имена, создаем простые
                    feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
                
                df_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
                df_train_processed['selling_price'] = y_train.values
                
                df_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
                df_test_processed['selling_price'] = y_test.values
                
                st.success("Предобработка данных выполнена успешно")
                
            else:
                st.error("Пайплайн не имеет ожидаемой структуры")
                df_train_processed = None
                df_test_processed = None
                
        except Exception as e:
            st.error(f"Ошибка при предобработке: {e}")
            df_train_processed = None
            df_test_processed = None
    
    if df_train_processed is not None:
        # Показываем информацию о данных
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Тренировочные данные")
            st.write(f"Количество строк: {df_train_processed.shape[0]}")
            st.write(f"Количество признаков: {df_train_processed.shape[1]}")
            st.write("Первые 5 строк:")
            st.dataframe(df_train_processed.head())
        
        with col2:
            st.subheader("Тестовые данные")
            st.write(f"Количество строк: {df_test_processed.shape[0]}")
            st.write(f"Количество признаков: {df_test_processed.shape[1]}")
            st.write("Первые 5 строк:")
            st.dataframe(df_test_processed.head())
        
        # Графики для тренировочных данных
        st.subheader("Графики по тренировочным данным")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Распределение цен")
            fig, ax = plt.subplots()
            df_train_processed['selling_price'].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_xlabel("Цена (руб)")
            ax.set_ylabel("Количество")
            ax.set_title("Распределение цен (тренировочные данные)")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.write("Сравнение распределений цен")
            fig, ax = plt.subplots()
            
            # Берем логарифм цен для лучшей визуализации
            train_log_price = np.log1p(df_train_processed['selling_price'])
            test_log_price = np.log1p(df_test_processed['selling_price'])
            
            ax.hist(train_log_price, bins=30, alpha=0.7, label='Тренировочные', edgecolor='black')
            ax.hist(test_log_price, bins=30, alpha=0.7, label='Тестовые', edgecolor='black')
            
            ax.set_xlabel("Логарифм цены")
            ax.set_ylabel("Количество")
            ax.set_title("Сравнение распределений цен")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        # Статистика по числовым признакам
        st.subheader("Статистика числовых признаков")
        
        # Выбираем первые 5 числовых признаков для анализа
        numeric_cols = df_train_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 5:
            display_cols = numeric_cols[:5]
        else:
            display_cols = numeric_cols
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Тренировочные данные:")
            st.write(df_train_processed[display_cols].describe())
        
        with col2:
            st.write("Тестовые данные:")
            st.write(df_test_processed[display_cols].describe())
        
        # Матрица корреляций для тренировочных данных
        if len(numeric_cols) > 1:
            st.subheader("Матрица корреляций (тренировочные данные)")
            
            # Берем только первые 10 признаков для читаемости
            if len(numeric_cols) > 10:
                corr_cols = numeric_cols[:10]
            else:
                corr_cols = numeric_cols
            
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df_train_processed[corr_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, square=True, cbar_kws={"shrink": 0.8})
            ax.set_title("Корреляция между признаками")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
        
        # Информация о пропущенных значениях
        st.subheader("Информация о пропущенных значениях")
        
        missing_train = df_train_processed.isnull().sum()
        missing_test = df_test_processed.isnull().sum()
        
        missing_df = pd.DataFrame({
            'Тренировочные_пропуски': missing_train,
            'Тестовые_пропуски': missing_test
        })
        
        st.write("Пропущенные значения после предобработки:")
        st.dataframe(missing_df[missing_df.sum(axis=1) > 0])

else:
    st.warning("Не удалось загрузить данные или пайплайн")

# Разделы предсказания и визуализации весов остаются без изменений...
# [Остальной код предсказания и визуализации весов из предыдущего варианта]

st.header('Предсказание цены')

pred_mode = st.radio("Способ ввода:", ["CSV файл", "Ручной ввод"], horizontal=True)

if pipeline is not None:
    if pred_mode == "CSV файл":
        pred_file = st.file_uploader("Загрузите CSV для предсказания", type=['csv'], key='pred')
        
        if pred_file is not None:
            df_pred = pd.read_csv(pred_file)
            st.write(f"Загружено {len(df_pred)} автомобилей")
            
            if st.button("Предсказать все цены"):
                with st.spinner("Выполняется предсказание..."):
                    try:
                        predictions = pipeline.predict(df_pred)
                        df_pred['predicted_price'] = predictions
                        
                        st.success(f"Предсказано {len(predictions)} цен")
                        
                        st.subheader("Результаты предсказания")
                        st.dataframe(df_pred[['predicted_price']].style.format({'predicted_price': '{:,.0f}'}))
                        
                        csv = df_pred.to_csv(index=False)
                        st.download_button(
                            "Скачать результаты",
                            csv,
                            "car_predictions.csv",
                            "text/csv"
                        )
                    except Exception as e:
                        st.error(f"Ошибка: {str(e)}")
    
    else:
        st.subheader("Введите характеристики автомобиля")
        
        left, right = st.columns(2)
        
        with left:
            name = st.text_input("Марка и модель", "Maruti Swift VDI")
            year = st.slider("Год выпуска", 1990, 2024, 2018)
            km_driven = st.number_input("Пробег (км)", 0, 500000, 50000, step=1000)
            fuel = st.selectbox("Топливо", ["Diesel", "Petrol", "CNG", "LPG"])
            seller_type = st.selectbox("Продавец", ["Individual", "Dealer", "Trustmark Dealer"])
        
        with right:
            transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
            owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
            mileage = st.number_input("Расход топлива", 5.0, 30.0, 18.5, step=0.1)
            engine = st.number_input("Объем двигателя (CC)", 600, 5000, 1200, step=100)
            max_power = st.number_input("Мощность (bhp)", 30.0, 500.0, 80.0, step=5.0)
            seats = st.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        
        if st.button("Узнать цену", type="primary"):
            input_data = pd.DataFrame([{
                'name': name,
                'year': year,
                'km_driven': km_driven,
                'fuel': fuel,
                'seller_type': seller_type,
                'transmission': transmission,
                'owner': owner,
                'mileage': mileage,
                'engine': engine,
                'max_power': max_power,
                'seats': seats
            }])
            
            try:
                prediction = pipeline.predict(input_data)[0]
                
                st.markdown("---")
                st.subheader("Результат предсказания")
                st.markdown(f"**Предсказанная цена:** {prediction:,.0f} руб.")
                st.write(f"Точность модели: R² = 0.69")
            except Exception as e:
                st.error(f"Ошибка предсказания: {str(e)}")

# Визуализация весов модели
st.header('Важность признаков модели')

if pipeline is not None and st.button("Показать важность признаков"):
    try:
        if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
            model = pipeline.named_steps['model']
            
            if hasattr(model, 'coef_'):
                coef = model.coef_
                
                # Получаем имена признаков
                feature_names = []
                if hasattr(pipeline, 'get_feature_names_out'):
                    feature_names = list(pipeline.get_feature_names_out())
                elif hasattr(pipeline.named_steps.get('preprocessor', None), 'get_feature_names_out'):
                    feature_names = list(pipeline.named_steps['preprocessor'].get_feature_names_out())
                else:
                    feature_names = [f"Признак {i+1}" for i in range(len(coef))]
                
                # Создаем DataFrame с весами
                weights_df = pd.DataFrame({
                    'Признак': feature_names,
                    'Вес': coef,
                    'Абсолютное_значение': np.abs(coef)
                }).sort_values('Абсолютное_значение', ascending=False)
                
                # Топ-15 признаков для визуализации
                top_weights = weights_df.head(15)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(top_weights))
                colors = ['green' if x > 0 else 'red' for x in top_weights['Вес']]
                
                ax.barh(y_pos, top_weights['Вес'], color=colors, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_weights['Признак'], fontsize=9)
                ax.set_xlabel("Вес признака")
                ax.set_title("Топ-15 самых важных признаков")
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                ax.grid(axis='x', alpha=0.3)
                
                st.pyplot(fig)
                
                # Таблица с весами
                st.subheader("Таблица весов признаков")
                st.dataframe(weights_df.head(20))
                
            else:
                st.write("Модель не имеет атрибута coef_")
        else:
            st.write("Не удалось извлечь модель из пайплайна")
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")

with st.sidebar:
    st.header("Информация")
    st.write("""
    **Описание модели:**
    - Алгоритм: Ridge регрессия
    - Точность (R²): 0.69
    - 31% предсказаний в пределах 10% от реальной цены
    
    **Данные для EDA:**
    Исходные данные после применения предобработки
    """)
