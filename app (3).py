import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import openai
import os
import pandas as pd
import matplotlib.pyplot as plt

API ключ из секрета
openai.api_key = st.secrets["OPENAI_API_KEY"]

Интерфейс чата
st.title("Chat с GPT")

user_input = st.text_input("Введите сообщение:")

if user_input:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    st.write(response.choices[0].message.content)

from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
import openai
import json
import io
import time
import warnings
from pandas.api.types import is_datetime64_any_dtype

warnings.filterwarnings('ignore')

# Настройки страницы
st.set_page_config(
    page_title="🤖 AI Data Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка API ключа OpenAI из Streamlit Secrets
if 'OPENAI_API_KEY' in st.secrets:
    openai.api_key = st.secrets['OPENAI_API_KEY']
else:
    openai.api_key = ""

# Тема приложения
theme = st.sidebar.radio("🎨 Тема", ["Светлая", "Темная"])
if theme == "Темная":
    st.markdown("""
        <style>
            .stApp { background-color: #1E1E1E; }
            .st-bb { background-color: transparent; }
            .st-at { background-color: #2E2E2E; }
            .css-1d391kg { color: white; }
            footer { visibility: hidden; }
        </style>
    """, unsafe_allow_html=True)

# Заголовок
st.title("🤖 AI Data Analyzer Pro")
st.markdown("""
    <div style="background-color:#f0f2f6;padding:10px;border-radius:10px;margin-bottom:20px;">
    <p style="color:#333;font-size:18px;">🚀 <b>Автоматический анализ данных с AI-powered инсайтами</b></p>
    <p style="color:#666;">Загрузите CSV, Excel или JSON — получите полный анализ и визуализацию</p>
    </div>
""", unsafe_allow_html=True)

# Кэшированная функция загрузки данных
@st.cache_data(show_spinner="Загружаю данные... ⏳", ttl=3600, max_entries=3)
def load_data(uploaded_file):
    try:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(io.BytesIO(file_bytes), encoding_errors='ignore')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(io.BytesIO(file_bytes))
        elif uploaded_file.name.endswith('.json'):
            data = json.loads(file_bytes.decode('utf-8'))
            return pd.json_normalize(data)
    except Exception as e:
        st.error(f"Ошибка загрузки: {str(e)}")
        return None

# Оптимизация памяти
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
    st.sidebar.info(f"Оптимизация памяти: {start_mem:.2f} MB → {end_mem:.2f} MB (сэкономлено {100*(start_mem-end_mem)/start_mem:.1f}%)")
    return df

# AI анализ данных (описание и статистика)
@st.cache_data(show_spinner="Анализирую данные... 🔍", ttl=600)
def analyze_with_ai(df):
    try:
        analysis = f"### 📊 Общий обзор данных\n"
        analysis += f"- **Строки:** {df.shape[0]}\n"
        analysis += f"- **Колонки:** {df.shape[1]}\n"
        analysis += f"- **Объем данных:** {df.memory_usage().sum() / 1024**2:.2f} MB\n\n"

        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            analysis += "### 🔢 Числовые данные\n"
            stats = df[num_cols].describe().transpose()
            stats['skew'] = df[num_cols].skew()
            analysis += stats[['mean', 'std', 'min', '50%', 'max', 'skew']].to_markdown()

        cat_cols = df.select_dtypes(exclude=np.number).columns
        if len(cat_cols) > 0:
            analysis += "\n\n### 🔤 Категориальные данные\n"
            for col in cat_cols:
                analysis += f"- **{col}**: {df[col].nunique()} уникальных значений\n"

        missing = df.isnull().sum()
        if missing.sum() > 0:
            analysis += "\n\n### ⚠️ Пропущенные значения\n"
            missing_percent = missing[missing > 0] / len(df) * 100
            missing_df = pd.DataFrame({'Колонка': missing_percent.index,
                                      'Пропуски': missing[missing > 0],
                                      '%': missing_percent.values.round(1)})
            analysis += missing_df.to_markdown(index=False)

        if len(num_cols) > 1:
            corr = df[num_cols].corr().abs().unstack().sort_values(ascending=False)
            strong_corr = corr[(corr > 0.7) & (corr < 1)].drop_duplicates()
            if len(strong_corr) > 0:
                analysis += "\n\n### 🔗 Сильные корреляции\n"
                for pair, value in strong_corr.items():
                    analysis += f"- {pair[0]} и {pair[1]}: {value:.2f}\n"

        return analysis
    except Exception as e:
        return f"Ошибка анализа: {str(e)}"

# Обнаружение аномалий IsolationForest
@st.cache_data(show_spinner="Ищу аномалии... 🕵️", ttl=300)
def detect_anomalies(df, column):
    try:
        if len(df) > 10000:
            sample = df.sample(min(5000, len(df)))
        else:
            sample = df

        model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        model.fit(sample[[column]])
        df['anomaly'] = model.predict(df[[column]])
        anomalies = df[df['anomaly'] == -1]
        return anomalies
    except:
        return None

# Анализ временных рядов с декомпозицией
@st.cache_data(show_spinner="Анализирую временные ряды... ⏳", ttl=300)
def time_series_analysis(df, date_col, value_col):
    try:
        df = df.set_index(date_col).sort_index()
        if len(df) > 1000:
            df = df.resample('D').mean()

        decomposition = seasonal_decompose(df[value_col], period=min(12, len(df)//2))

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=df.index, y=df[value_col], name='Исходные данные'), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Тренд'), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Сезонность'), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Остатки'), row=4, col=1)

        fig.update_layout(height=800, title_text="Декомпозиция временного ряда")
        return fig
    except:
        return None

# Генерация инсайтов через OpenAI GPT
@st.cache_data(show_spinner="Генерирую AI инсайты... 🤖", ttl=600)
def generate_ai_insights(df):
    try:
        if not openai.api_key:
            return "🔑 Ключ OpenAI API не установлен. Добавьте его в Secrets."

        prompt = (
            f"Дай краткий аналитический отчет и рекомендации по данным.\n"
            f"Данные имеют {df.shape[0]} строк и {df.shape[1]} колонок.\n"
            f"Колонки: {list(df.columns)}.\n"
            f"Первые 5 строк:\n{df.head().to_dict()}\n"
            f"Пожалуйста, дай инсайты и рекомендации."
        )

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты аналитик данных."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        text = response['choices'][0]['message']['content']
        return text
    except Exception as e:
        return f"Ошибка OpenAI: {str(e)}"

# Визуализация данных
def create_visualization(df, viz_type, x=None, y=None, z=None, color=None, size=None):
    try:
        viz_df = df.sample(min(10000, len(df))) if len(df) > 10000 else df

        if viz_type == "Гистограмма":
            fig = px.histogram(viz_df, x=x, color=color, marginal="box", nbins=50)

        elif viz_type == "Тепловая карта":
            corr = viz_df.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')

        elif viz_type == "3D Scatter":
            fig = px.scatter_3d(viz_df, x=x, y=y, z=z, color=color, size=size)

        elif viz_type == "Аномалии":
            anomalies = detect_anomalies(df, x)
            fig = px.scatter(viz_df, x=x, y=y, color=viz_df.index.isin(anomalies.index) if anomalies is not None else None)

        elif viz_type == "Временной ряд":
            if len(viz_df) > 1000 and is_datetime64_any_dtype(viz_df[x]):
                viz_df = viz_df.set_index(x).resample('D').mean().reset_index()
            fig = px.line(viz_df, x=x, y=y, color=color)
            if len(viz_df) > 30:
                viz_df['rolling'] = viz_df[y].rolling(7).mean()
                fig.add_trace(go.Scatter(x=viz_df[x], y=viz_df['rolling'], mode='lines', name='Скользящее среднее (7)'))

        elif viz_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=viz_df[x],
                open=viz_df[y],
                high=viz_df[y] + viz_df[y].std(),
                low=viz_df[y] - viz_df[y].std(),
                close=viz_df[y])])

        else:
            fig = px.scatter(viz_df, x=x, y=y, color=color, size=size)

        fig.update_layout(
            template="plotly_dark" if theme == "Темная" else "plotly_white",
            hovermode="x unified",
            height=600
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))

        return fig
    except Exception as e:
        st.error(f"Ошибка визуализации: {e}")
        return None

# === UI ===

uploaded_file = st.file_uploader("📂 Загрузите CSV, Excel или JSON файл", type=['csv', 'xlsx', 'xls', 'json'])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None and not df.empty:
        df = reduce_mem_usage(df)

        st.sidebar.subheader("Выберите колонки для анализа и визуализации")
        columns = df.columns.tolist()

        x_axis = st.sidebar.selectbox("X-ось", options=columns)
        y_axis = st.sidebar.selectbox("Y-ось", options=[None] + columns)
        z_axis = st.sidebar.selectbox("Z-ось (для 3D)", options=[None] + columns)
        color = st.sidebar.selectbox("Цвет", options=[None] + columns)
        size = st.sidebar.selectbox("Размер", options=[None] + columns)

        viz_type = st.sidebar.selectbox("Тип визуализации", [
            "Гистограмма",
            "Тепловая карта",
            "3D Scatter",
            "Временной ряд",
            "Candlestick",
            "Аномалии",
            "Точечная диаграмма"
        ])

        st.sidebar.markdown("---")
        st.sidebar.subheader("AI анализ данных")
        if st.sidebar.button("Анализировать данные"):
            with st.spinner("Анализирую..."):
                analysis_text = analyze_with_ai(df)
                st.markdown(analysis_text)

        st.sidebar.subheader("AI инсайты и рекомендации")
        if st.sidebar.button("Получить AI инсайты"):
            with st.spinner("Генерирую инсайты..."):
                insights = generate_ai_insights(df)
                st.markdown(insights)

        if st.sidebar.button("Показать визуализацию"):
            fig = create_visualization(df, viz_type, x_axis, y_axis, z_axis, color, size)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        st.sidebar.markdown("---")
        if st.sidebar.button("Показать исходные данные"):
            st.dataframe(df)

    else:
        st.warning("Не удалось загрузить данные из файла или файл пустой.")
else:
    st.info("Пожалуйста, загрузите файл для начала анализа.")
