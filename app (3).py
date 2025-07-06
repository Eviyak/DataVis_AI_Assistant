import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import openai
import io
import json
import warnings
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
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

# Тема дневная (без выбора)
st.markdown("""
    <style>
        .stApp { background-color: #f0f2f6; color: #000000; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# Заголовок
st.title("🤖 AI Data Analyzer Pro")
st.markdown("""
    <div style="background-color:#ffffff;padding:10px;border-radius:10px;margin-bottom:20px;">
    <p style="color:#333;font-size:18px;">🚀 <b>Автоматический анализ данных с AI-powered инсайтами</b></p>
    <p style="color:#666;">Загрузите CSV, Excel или JSON — получите полный анализ и визуализацию</p>
    </div>
""", unsafe_allow_html=True)

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

@st.cache_data(show_spinner="Генерирую AI инсайты... 🤖", ttl=600)
def generate_ai_insights(df):
    if not openai.api_key:
        return "🔑 Ключ OpenAI API не установлен. Добавьте его в Secrets."

    prompt = (
        f"Ты аналитик данных. Сделай краткий аналитический отчет по данным.\n"
        f"Данные: {df.shape[0]} строк, {df.shape[1]} колонок.\n"
        f"Колонки: {list(df.columns)}.\n"
        f"Первые 5 строк:\n{df.head().to_dict()}\n\n"
        f"Дай краткие инсайты и рекомендации по данным."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты аналитик данных."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Ошибка вызова OpenAI API: {str(e)}"

@st.cache_data(show_spinner="Генерирую рекомендации по визуализациям... 🎨", ttl=600)
def generate_viz_recommendations(df):
    if not openai.api_key:
        return None

    prompt = (
        f"Ты эксперт по визуализации данных. На основе этих данных предложи несколько типов визуализаций с выбором колонок:\n"
        f"Данные: {df.shape[0]} строк, {df.shape[1]} колонок.\n"
        f"Колонки: {list(df.columns)}.\n"
        f"Первые 5 строк:\n{df.head().to_dict()}\n\n"
        f"Предложи до 3 визуализаций в формате JSON с полями:\n"
        f'{{"viz_type": "...", "x_axis": "...", "y_axis": "...", "z_axis": "...", "color": "...", "size": "..."}}\n'
        f"Типы визуализаций могут быть: гистограмма, тепловая карта, scatter, 3D scatter, временной ряд, candlestick, ящик с усами и др."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты эксперт по визуализации данных."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        text = response['choices'][0]['message']['content']
        viz_recs = clean_json(text)
        return viz_recs
    except Exception:
        return None


def create_visualization(df, viz_type, x_axis=None, y_axis=None, z_axis=None, color=None, size=None):
    try:
        if viz_type is None:
            return None
        viz_type = viz_type.lower()

        if viz_type == "гистограмма":
            if x_axis:
                fig = px.histogram(df, x=x_axis, color=color, nbins=30)
                return fig
        elif viz_type == "тепловая карта":
            if x_axis and y_axis:
                pivot = pd.pivot_table(df, values=size or y_axis, index=y_axis, columns=x_axis, aggfunc='mean')
                fig = px.imshow(pivot)
                return fig
        elif viz_type == "scatter":
            if x_axis and y_axis:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color, size=size)
                return fig
        elif viz_type == "3d scatter":
            if x_axis and y_axis and z_axis:
                fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=color, size=size)
                return fig
        elif viz_type == "временной ряд":
            if x_axis and y_axis:
                fig = px.line(df, x=x_axis, y=y_axis, color=color)
                return fig
        elif viz_type == "ящик с усами":
            if x_axis and y_axis:
                fig = px.box(df, x=x_axis, y=y_axis, color=color)
                return fig
        elif viz_type == "candlestick":
            # Пример: x_axis - дата, y_axis - open, color - high, size - low, z_axis - close
            required_cols = [x_axis, y_axis, color, size]
            if all(c in df.columns for c in [x_axis, y_axis, color, size] if c):
                fig = go.Figure(data=[go.Candlestick(
                    x=df[x_axis],
                    open=df[y_axis],
                    high=df[color],
                    low=df[size],
                    close=df[z_axis] if z_axis in df.columns else df[y_axis]
                )])
                return fig
        return None
    except Exception:
        return None



### --- Streamlit UI ---

st.sidebar.header("Загрузите файл с данными")
uploaded_file = st.sidebar.file_uploader("CSV, Excel или JSON", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        df = reduce_mem_usage(df)
        st.success(f"Файл загружен: {uploaded_file.name} ({df.shape[0]} строк, {df.shape[1]} колонок)")
        st.dataframe(df.head())

        st.subheader("🤖 AI Инсайты по данным")
        insights = generate_ai_insights(df)
        st.markdown(insights)

        st.subheader("🎨 Рекомендации по визуализациям")
        viz_recs = generate_viz_recommendations(df)
        if viz_recs:
            if isinstance(viz_recs, dict):
                viz_recs = [viz_recs]  # если один объект, обернем в список
            for i, viz in enumerate(viz_recs):
                st.markdown(f"**Визуализация {i+1}:** {viz.get('viz_type', 'Не указано')}")
                fig = create_visualization(
                    df,
                    viz.get('viz_type'),
                    viz.get('x_axis'),
                    viz.get('y_axis'),
                    viz.get('z_axis'),
                    viz.get('color'),
                    viz.get('size')
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Невозможно построить эту визуализацию с текущими данными.")
        else:
            st.info("Нет рекомендаций по визуализациям.")

else:
    st.info("Пожалуйста, загрузите файл для анализа.")

