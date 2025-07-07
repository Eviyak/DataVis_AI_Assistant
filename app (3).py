import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import io
import json
import warnings
import joblib
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

st.set_page_config(page_title="DataJournalist ML Assistant", page_icon="📰", layout="wide")

if 'OPENAI_API_KEY' in st.secrets:
    openai.api_key = st.secrets['OPENAI_API_KEY']
else:
    openai.api_key = ""

# ==== СТИЛИ ====
st.markdown("""
    <style>
        .stApp { background-color: #f0f2f6; color: #000000; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

st.title("📰 DataJournalist ML Assistant")
st.markdown("""
    <div style="background-color:#ffffff;padding:20px;border-radius:10px;margin-bottom:20px;">
    <h3 style="color:#333;">🚀 Автоматический анализ и визуализация данных + ML</h3>
    <p style="color:#666;">Загрузите данные → получите инсайты, модели, визуализации</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_data
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    st.sidebar.info(f"Оптимизация памяти: {start_mem:.2f} MB → {end_mem:.2f} MB")
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
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return df
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(df[num_cols])
    df['anomaly'] = preds
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
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
def generate_shap_plot(model, X_test, feature_names):
    import shap
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    fig = shap.plots.beeswarm(shap_values, show=False)
    return fig

def generate_ai_report(df, model, problem_type, target, metrics):
    if not openai.api_key:
        return "🔑 Ключ OpenAI API не установлен."
    prompt = (
        f"Ты аналитик данных. Напиши короткий журналистский отчет по результатам машинного обучения.\n\n"
        f"Тип задачи: {problem_type}\n"
        f"Целевая переменная: {target}\n"
        f"Метрики: {metrics}\n"
        f"Колонки: {list(df.columns)}\n"
        f"Примеры данных:\n{df.head(3).to_dict()}\n\n"
        f"Напиши отчет в формате:\n1. Введение\n2. Основные выводы\n3. Что важно рассказать читателю\n"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ты журналист-аналитик, пишущий аналитические тексты."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Ошибка AI отчета: {e}"

def generate_flourish_recommendations(df, target):
    prompt = (
        f"Ты журналист и визуализатор данных. По следующему датафрейму:\n"
        f"Колонки: {list(df.columns)}\n"
        f"Целевая переменная: {target}\n\n"
        f"Предложи 3 идеи для визуализации в стиле инфографики для публикации, как на сайте РБК или BBC. "
        f"Формат ответа: короткие тезисы, без кода, на русском языке."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ты специалист по визуализации данных и журналистике."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Ошибка AI рекомендации: {e}"

# === UI ===
st.sidebar.header("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Выберите CSV, Excel или JSON файл", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    df_raw, error = load_data(uploaded_file)
    if error:
        st.error(error)
    else:
        df_clean = fill_missing_values(df_raw)
        df_clean = mark_anomalies(df_clean)
        df_clean = reduce_mem_usage(df_clean)

        st.success(f"✅ Файл успешно загружен: {uploaded_file.name}")
        st.write("### 📄 Пример данных:")
        st.dataframe(df_clean.head(10), use_container_width=True)

        # Скачать очищенные данные
        csv_clean = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Скачать очищенные данные", data=csv_clean, file_name="cleaned_data.csv", mime="text/csv")

        # Выбор задачи
        st.sidebar.header("🧠 ML-задача")
        ml_task = st.sidebar.selectbox("Что вы хотите сделать?", ["Прогнозирование (регрессия)", "Классификация", "Кластеризация"])

        if ml_task in ["Прогнозирование (регрессия)", "Классификация"]:
            target = st.sidebar.selectbox("Целевая переменная", options=df_clean.columns)
            if st.sidebar.button("🚀 Обучить модель"):
                X, y, scaler = prepare_data_for_ml(df_clean, target)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if ml_task == "Прогнозирование (регрессия)":
                    model = RandomForestRegressor(random_state=42)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    rmse = mean_squared_error(y_test, preds, squared=False)
                    metrics = {"RMSE": rmse}
                    problem_type = "regression"

                else:
                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    metrics = {"Accuracy": acc}
                    problem_type = "classification"

                st.session_state.update({
                    'model': model,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': preds,
                    'metrics': metrics,
                    'problem_type': problem_type,
                    'feature_names': df_clean.drop(columns=[target]).columns,
                    'df': df_clean,
                    'target': target
                })
                st.experimental_rerun()

        elif ml_task == "Кластеризация":
            num_cols = df_clean.select_dtypes(include=np.number).columns
            n_clusters = st.sidebar.slider("Количество кластеров", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_clustered = df_clean.copy()
            df_clustered["Cluster"] = kmeans.fit_predict(df_clustered[num_cols])
            cluster_means = df_clustered.groupby("Cluster")[num_cols].mean().round(2)

            st.session_state.update({
                'df_clustered': df_clustered,
                'cluster_analysis': cluster_means,
                'model': kmeans
            })
            st.experimental_rerun()
# Визуальные вкладки
if 'model' in st.session_state or 'df_clustered' in st.session_state:
    tabs = st.tabs(["📊 Результаты", "📈 Визуализация", "📝 Журналистский отчет", "⚙️ Настройки"])

    # === Результаты ===
    with tabs[0]:
        st.subheader("📊 Результаты модели")
        if st.session_state.get('problem_type') == "classification":
            st.write("**Метрики:**")
            st.json(st.session_state['metrics'])

            cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
            st.write("**Матрица ошибок:**")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            st.pyplot(fig)

        elif st.session_state.get('problem_type') == "regression":
            st.write("**RMSE:**", round(st.session_state['metrics']['RMSE'], 3))

        elif 'df_clustered' in st.session_state:
            st.write("**Средние значения по кластерам:**")
            st.dataframe(st.session_state['cluster_analysis'])

    # === Визуализация ===
    with tabs[1]:
        st.subheader("📈 Простая визуализация")
        df_vis = st.session_state.get('df', pd.DataFrame())
        if not df_vis.empty:
            x = st.selectbox("Ось X", options=df_vis.columns)
            y = st.selectbox("Ось Y", options=df_vis.columns)
            chart = px.scatter(df_vis, x=x, y=y, color_discrete_sequence=["#1f77b4"])
            st.plotly_chart(chart, use_container_width=True)

    # === Журналистский отчет ===
    with tabs[2]:
        st.subheader("📝 Генерация журналистского отчета")
        report = generate_ai_report(
            st.session_state['df'],
            st.session_state['model'],
            st.session_state['problem_type'],
            st.session_state['target'],
            st.session_state['metrics']
        )
        st.markdown(report)

        st.subheader("🎯 Идеи для визуализаций в Flourish")
        flourish_tips = generate_flourish_recommendations(st.session_state['df'], st.session_state['target'])
        st.markdown(flourish_tips)

    # === Настройки / Дополнительно ===
    with tabs[3]:
        st.subheader("⚙️ Настройки и информация")
        st.write("Версия: 1.0.0")
        st.write("Автор: DataJournalist Assistant by GPT")
        st.markdown("GitHub: [ссылка-здесь](https://github.com/your-project)")
        st.markdown("Если вы заметили ошибку — обратитесь к разработчику.")
