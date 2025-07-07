import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import io
import json
import warnings
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

warnings.filterwarnings('ignore')

# Настройки страницы
st.set_page_config(
    page_title="DataJournalist ML Assistant",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка API ключа OpenAI
if 'OPENAI_API_KEY' in st.secrets:
    openai.api_key = st.secrets['OPENAI_API_KEY']
else:
    openai.api_key = st.sidebar.text_input("Введите OpenAI API ключ:", type="password")

# Стилизация
st.markdown("""
    <style>
        .stApp { background-color: #f0f2f6; color: #000000; }
        footer { visibility: hidden; }
        .stProgress > div > div > div > div { background: linear-gradient(to right, #ff4b4b, #ff9a9e); }
        .stButton>button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

st.title("📰 DataJournalist ML Assistant")
st.markdown("""
    <div style="background-color:#ffffff;padding:20px;border-radius:10px;margin-bottom:20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);">
    <h3 style="color:#333;margin-top:0;">🚀 Автоматизированный анализ данных и ML для журналистов</h3>
    <p style="color:#666;">Загрузите данные → Выберите задачу → Получите готовую модель и инсайты</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Загружаю данные... ⏳", ttl=3600, max_entries=3)
def load_data(uploaded_file):
    try:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(io.BytesIO(file_bytes)), None
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(io.BytesIO(file_bytes)), None
        elif uploaded_file.name.endswith('.json'):
            data = json.loads(file_bytes.decode('utf-8'))
            return pd.json_normalize(data), None
    except Exception as e:
        return None, f"Ошибка загрузки: {str(e)}"

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                # Добавляем проверку на наличие числовых значений
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float16)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)
                    except:
                        # Если возникает ошибка сравнения, оставляем исходный тип
                        df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    st.sidebar.info(f"Оптимизация памяти: {start_mem:.2f} MB → {end_mem:.2f} MB (сэкономлено {100*(start_mem-end_mem)/start_mem:.1f}%)")
    return df

def fill_missing_values(df):
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].isnull().sum() > 0:
            if is_numeric_dtype(df_filled[col]):
                df_filled[col].fillna(df_filled[col].median(), inplace=True)
            else:
                mode_val = df_filled[col].mode()
                if not mode_val.empty:
                    df_filled[col].fillna(mode_val[0], inplace=True)
                else:
                    df_filled[col].fillna("Unknown", inplace=True)
    return df_filled

def mark_anomalies(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(num_cols) == 0:
        return df
    
    df_processed = df.copy()
    
    X = df_processed[num_cols].values
    X = np.nan_to_num(X)
    
    try:
        # Инициализируем и обучаем модель
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(X)
        
        # Добавляем метки аномалий в DataFrame
        df_processed['anomaly'] = preds
        df_processed['anomaly'] = df_processed['anomaly'].map({1: 0, -1: 1})
        
        return df_processed
    except Exception as e:
        st.error(f"Ошибка при обнаружении аномалий: {str(e)}")
        return df

def prepare_data_for_ml(df, target_column):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X, y, problem_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cm = None
    
    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model_name = "Random Forest (Классификация)"
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        metrics = {"Точность": accuracy}
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_name = "Random Forest (Регрессия)"
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        metrics = {"RMSE": rmse, "MSE": mse}
    
    return model, metrics, X_test, y_test, y_pred, cm

def generate_shap_plot(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    
    plt.tight_layout()
    return plt.gcf()

def generate_ai_report(df, model, problem_type, target, metrics):
    if not openai.api_key:
        return "🔑 Ключ OpenAI API не установлен. Добавьте его в настройках."
    
    prompt = f"""
Ты - журналист-аналитик с опытом в data science. Подготовь отчет о результатах анализа данных и построенной модели машинного обучения.

Данные:
- Количество наблюдений: {df.shape[0]}
- Количество признаков: {df.shape[1]}
- Целевая переменная: {target}
- Тип задачи: {'Классификация' if problem_type == 'classification' else 'Регрессия'}

Метрики модели:
{json.dumps(metrics, indent=2)}

Важные переменные (первые 5):
{df.columns.tolist()[:5]}

Сгенерируй:
1. Простое объяснение что делает модель
2. Ключевые инсайты о важных признаках
3. Как журналист может использовать эти результаты в статье
4. Ограничения анализа
5. Рекомендации по дальнейшему исследованию

Пиши кратко, понятно, без технического жаргона. Используй маркированные списки.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты журналист-аналитик, объясняющий сложные ML-концепты простым языком."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=1500
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Ошибка вызова OpenAI API: {str(e)}"

def generate_flourish_recommendations(df, target):
    if not openai.api_key:
        return None
    
    prompt = f"""
На основе данных с колонками: {list(df.columns)} и целевой переменной '{target}', 
предложи 3 оптимальных типа визуализаций для Flourish. Для каждого укажи:

1. Тип визуализации
2. Какие колонки использовать
3. Почему это будет эффективно
4. Рекомендации по настройке во Flourish

Пример ответа:
- **Тип**: Интерактивная карта
  **Колонки**: Регион, {target}
  **Обоснование**: Позволяет показать географическое распределение показателя
  **Настройки**: Использовать российские регионы в формате GeoJSON
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты эксперт по визуализации данных для журналистики."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Ошибка OpenAI API: {e}"

def cluster_data(df, n_clusters):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        return df, "Нет числовых колонок для кластеризации"
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    df['Cluster'] = clusters
    cluster_analysis = df.groupby('Cluster')[numeric_cols].mean().reset_index()
    
    return df, cluster_analysis

def show_results_tab():
    st.subheader("Результаты анализа")
    
    if ml_task in ["Прогнозирование (регрессия)", "Классификация"]:
        st.write("### Метрики модели")
        for metric, value in st.session_state['metrics'].items():
            st.metric(label=metric, value=f"{value:.4f}")
        
        if st.session_state['problem_type'] == "classification" and st.session_state['cm'] is not None:
            st.write("### Матрица ошибок")
            try:
                # Получаем уникальные классы из тестовых и предсказанных значений
                all_classes = np.unique(np.concatenate([
                    st.session_state['y_test'], 
                    st.session_state['y_pred']
                ]))
                
                # Создаем фигуру для матрицы ошибок
                fig, ax = plt.subplots(figsize=(8, 6))
                ConfusionMatrixDisplay.from_predictions(
                    st.session_state['y_test'],
                    st.session_state['y_pred'],
                    display_labels=all_classes,
                    cmap='Blues',
                    ax=ax,
                    values_format='d'
                )
                ax.set_title('Матрица ошибок')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Ошибка при построении матрицы ошибок: {str(e)}")
                
                # Альтернативное отображение матрицы ошибок
                try:
                    cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
                    unique_classes = np.unique(np.concatenate([
                        st.session_state['y_test'], 
                        st.session_state['y_pred']
                    ]))
                    
                    # Проверяем соответствие размеров матрицы и меток
                    if cm.shape[0] == len(unique_classes) and cm.shape[1] == len(unique_classes):
                        st.write(pd.DataFrame(
                            cm,
                            index=[f"Истинный {c}" for c in unique_classes],
                            columns=[f"Предсказанный {c}" for c in unique_classes]
                        ))
                    else:
                        st.write("Числовая матрица ошибок:", cm)
                except Exception as e2:
                    st.write("Не удалось отобразить матрицу ошибок:", str(e2))
    
    elif ml_task == "Кластеризация":
        st.write("### Распределение по кластерам")
        cluster_counts = st.session_state['df_clustered']['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)
        
        st.write("### Характеристики кластеров")
        st.dataframe(st.session_state['cluster_analysis'])
        
        if len(df_clean.select_dtypes(include=np.number).columns) >= 2:
            num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
            col1, col2 = st.selectbox("Выберите ось X", num_cols, index=0), st.selectbox("Выберите ось Y", num_cols, index=1)
            
            fig = px.scatter(
                st.session_state['df_clustered'],
                x=col1,
                y=col2,
                color='Cluster',
                hover_data=df_clean.columns.tolist(),
                title="Визуализация кластеров"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_visualizations_tab():
    st.subheader("Визуализация результатов")
    
    if ml_task in ["Прогнозирование (регрессия)", "Классификация"]:
        try:
            st.write("### Важность признаков (SHAP)")
            with st.spinner("Генерирую SHAP-визуализацию..."):
                fig = generate_shap_plot(
                    st.session_state['model'],
                    st.session_state['X_test'],
                    st.session_state['feature_names']
                )
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка при создании SHAP-визуализации: {str(e)}")
        
        if st.session_state['problem_type'] == "regression":
            st.write("### Прогнозы vs Фактические значения")
            results = pd.DataFrame({
                'Фактические': st.session_state['y_test'],
                'Прогнозные': st.session_state['y_pred']
            })
            
            try:
                fig = px.scatter(
                    results, 
                    x='Фактические', 
                    y='Прогнозные',
                    trendline='ols',
                    title="Сравнение прогнозов и фактических значений"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Не удалось построить трендлинию: {str(e)}. Показываю scatter plot без линии тренда.")
                fig = px.scatter(
                    results, 
                    x='Фактические', 
                    y='Прогнозные',
                    title="Сравнение прогнозов и фактических значений (без тренда)"
                )
                st.plotly_chart(fig, use_container_width=True)

def show_report_tab():
    st.subheader("Журналистский отчет")
    
    if ml_task in ["Прогнозирование (регрессия)", "Классификация"]:
        report = generate_ai_report(
            st.session_state['df'],
            st.session_state['model'],
            st.session_state['problem_type'],
            st.session_state['target'],
            st.session_state['metrics']
        )
        st.markdown(report)
        
        st.divider()
        
        st.write("### Рекомендации по визуализациям (Flourish)")
        flourish_recs = generate_flourish_recommendations(
            st.session_state['df'],
            st.session_state['target']
        )
        if flourish_recs:
            st.markdown(flourish_recs)
        else:
            st.warning("Не удалось сгенерировать рекомендации для Flourish")
    
    elif ml_task == "Кластеризация":
        st.write("### Интерпретация кластеров")
        cluster_summary = st.session_state['cluster_analysis'].to_dict()
        prompt = f"""
Проанализируй характеристики кластеров и предложи интерпретацию для журналиста:

Характеристики кластеров:
{cluster_summary}

Сгенерируй:
1. Краткое описание каждого кластера
2. Как можно назвать каждый кластер
3. Идеи для статей на основе кластерного анализа
"""
        try:
            if openai.api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Ты журналист-аналитик, специализирующийся на кластерном анализе."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                st.markdown(response['choices'][0]['message']['content'])
            else:
                st.warning("Для генерации отчета требуется OpenAI API ключ")
        except Exception as e:
            st.error(f"Ошибка OpenAI API: {e}")

def show_settings_tab():
    st.subheader("Настройки модели")
    
    if ml_task in ["Прогнозирование (регрессия)", "Классификация"]:
        # Создаем буфер в памяти для сохранения модели
        buffer = io.BytesIO()
        joblib.dump(st.session_state['model'], buffer)
        buffer.seek(0)
        
        st.download_button(
            label="💾 Скачать модель (joblib)",
            data=buffer,
            file_name=f"model_{datetime.now().strftime('%Y%m%d')}.joblib",
            mime="application/octet-stream"
        )
        
        st.write("### Тестовый прогноз")
        sample = df_clean.drop(columns=[st.session_state['target']]).iloc[0:1]
        st.write("Данные для прогноза:")
        st.dataframe(sample)
        
        if st.button("Сделать прогноз"):
            sample_prepared = prepare_data_for_ml(sample, st.session_state['target'])[0]
            prediction = st.session_state['model'].predict(sample_prepared)
            st.metric(label="Прогноз", value=prediction[0])

# Основной интерфейс
def main():
    st.sidebar.header("1. Загрузите данные")
    uploaded_file = st.sidebar.file_uploader("CSV, Excel или JSON", type=["csv", "xlsx", "xls", "json"])

    global df, df_clean, ml_task
    df = None
    df_clean = None

    if uploaded_file:
        df, error = load_data(uploaded_file)
        if df is not None:
            df = reduce_mem_usage(df)
            st.sidebar.success(f"Файл загружен: {uploaded_file.name}")
            
            with st.expander("🔍 Предварительный просмотр данных", expanded=True):
                st.dataframe(df.head(3))
                st.caption(f"Загружено {df.shape[0]} строк, {df.shape[1]} колонок")
            
            with st.spinner("🧹 Автоматически очищаю данные..."):
                df_clean = fill_missing_values(df)
                df_clean = mark_anomalies(df_clean)
            
            st.success("✅ Данные очищены! Добавлен столбец 'anomaly' для аномалий")
            
            col1, col2 = st.columns(2)
            with col1:
                csv = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать CSV для Flourish",
                    data=csv,
                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Оптимизированный формат для загрузки в Flourish"
                )
            with col2:
                json_data = df_clean.to_json(orient='records', force_ascii=False)
                st.download_button(
                    label="📥 Скачать JSON для Flourish",
                    data=json_data,
                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    help="Формат JSON для сложных визуализаций"
                )
            
            st.sidebar.header("2. Выберите задачу ML")
            ml_task = st.sidebar.selectbox("Тип задачи", 
                                         ["Прогнозирование (регрессия)", 
                                          "Классификация", 
                                          "Кластеризация"],
                                         index=0)
            
            st.sidebar.header("3. Настройте параметры")
            
            if ml_task in ["Прогнозирование (регрессия)", "Классификация"]:
                target_col = st.sidebar.selectbox("Выберите целевую переменную", df_clean.columns)
                
                if st.sidebar.button("▶ Обучить модель", type="primary"):
                    with st.spinner("🔄 Обучение модели..."):
                        problem_type = "regression" if ml_task == "Прогнозирование (регрессия)" else "classification"
                        
                        X, y, scaler = prepare_data_for_ml(df_clean, target_col)
                        model, metrics, X_test, y_test, y_pred, cm = train_model(X, y, problem_type)
                        
                        st.session_state['model'] = model
                        st.session_state['metrics'] = metrics
                        st.session_state['X_test'] = X_test
                        st.session_state['y_test'] = y_test
                        st.session_state['y_pred'] = y_pred
                        st.session_state['cm'] = cm
                        st.session_state['feature_names'] = df_clean.drop(columns=[target_col]).columns.tolist()
                        st.session_state['target'] = target_col
                        st.session_state['problem_type'] = problem_type
                        st.session_state['df'] = df_clean
                        
                        st.success("✅ Модель успешно обучена!")
            
            elif ml_task == "Кластеризация":
                n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 4)
                
                if st.sidebar.button("▶ Выполнить кластеризацию", type="primary"):
                    with st.spinner("🔍 Выполняю кластеризацию..."):
                        df_clustered, cluster_analysis = cluster_data(df_clean, n_clusters)
                        
                        st.session_state['df_clustered'] = df_clustered
                        st.session_state['cluster_analysis'] = cluster_analysis
                        
                        st.success(f"✅ Данные разбиты на {n_clusters} кластеров!")
            
            # Проверяем, есть ли что показывать во вкладках
            show_tabs = False
            if ml_task in ["Прогнозирование (регрессия)", "Классификация"] and 'model' in st.session_state:
                show_tabs = True
            elif ml_task == "Кластеризация" and 'df_clustered' in st.session_state:
                show_tabs = True
            
            if show_tabs:
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Результаты", "📈 Визуализации", "📝 Журналистский отчет", "⚙️ Настройки"])
                
                with tab1:
                    show_results_tab()
                
                with tab2:
                    show_visualizations_tab()
                
                with tab3:
                    show_report_tab()
                
                with tab4:
                    show_settings_tab()
            else:
                st.info("ℹ️ Нажмите 'Обучить модель' или 'Выполнить кластеризацию' чтобы увидеть результаты")
        
        else:
            st.error(f"Ошибка загрузки данных: {error}")
    else:
        st.info("👈 Пожалуйста, загрузите файл для начала анализа")
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80", 
                 caption="Инструмент для журналистских расследований на основе данных")

if __name__ == "__main__":
    main()
