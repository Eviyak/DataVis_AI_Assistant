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
    page_title="InsightBot Pro",
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
st.title("InsightBot Pro")
st.markdown("""
    <div style="background-color:#ffffff;padding:10px;border-radius:10px;margin-bottom:20px;">
    <p style="color:#333;font-size:18px;">🚀 <b>Автоматический анализ данных с AI-powered инсайтами</b></p>
    <p style="color:#666;">Загрузите CSV, Excel или JSON — получите полный анализ и визуализацию</p>
    </div>
""", unsafe_allow_html=True)

# === КЭШ ЗАГРУЗКИ ===
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
        analysis = ""
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

    prompt = f"""
Ты — эксперт по визуализации данных. Посмотри на колонки этих данных: {list(df.columns)}.
Предложи 3 простые и понятные рекомендации для построения графиков. 
Пиши по-русски и коротко. Например:

- Построй гистограмму для колонки 'Age'
- Построй scatter plot с 'Height' по оси X и 'Weight' по оси Y
- Построй box plot для колонки 'Salary'

Пиши в таком же формате, без JSON и лишних объяснений.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты эксперт по визуализации данных."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Ошибка OpenAI API: {e}"

# --- NEW: функция очистки данных ---
def clean_data(df):
    df_clean = df.copy()

    # Заполнение пропусков: числовые - медианой, категориальные - модой или "Unknown"
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
            else:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col].fillna(mode_val[0], inplace=True)
                else:
                    df_clean[col].fillna("Unknown", inplace=True)

    # Выделение аномалий через IsolationForest (только для числовых колонок)
    num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        iso_forest = IsolationForest(contamination=0.01, random_state=42)
        preds = iso_forest.fit_predict(df_clean[num_cols])
        df_clean['anomaly'] = preds
        df_clean['anomaly'] = df_clean['anomaly'].map({1: 'normal', -1: 'anomaly'})
    else:
        df_clean['anomaly'] = 'no numeric data'

    return df_clean

# === UI ===
st.sidebar.header("Загрузите файл с данными")
uploaded_file = st.sidebar.file_uploader("CSV, Excel или JSON", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        df = reduce_mem_usage(df)
        st.success(f"Файл загружен: {uploaded_file.name} ({df.shape[0]} строк, {df.shape[1]} колонок)")
        st.dataframe(df.head())

        # Автоматическая очистка данных (NEW)
        st.subheader("🧹 Очищенные данные (заполнение пропусков, пометка аномалий)")
        df_clean = clean_data(df)
        st.dataframe(df_clean.head())

        st.markdown(f"Количество строк с аномалиями: {(df_clean['anomaly'] == 'anomaly').sum()}")

        st.subheader("📊 Общий анализ данных")
        summary = analyze_with_ai(df_clean)  # анализируем очищенный датафрейм
        st.markdown(summary)

        st.subheader("🤖 AI Инсайты по данным")
        insights = generate_ai_insights(df_clean)
        st.markdown(insights)

        st.subheader("🎨 Рекомендации по визуализациям")
        viz_recs = generate_viz_recommendations(df_clean)
        if viz_recs:
            st.markdown(viz_recs)
        else:
            st.info("Нет рекомендаций по визуализациям.")

        # NEW: Кнопка скачивания очищенного файла
        csv_buffer = io.StringIO()
        df_clean.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Скачать очищенный CSV для Flourish и др.",
            data=csv_buffer.getvalue(),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    else:
        st.error("Не удалось загрузить данные из файла.")
else:
    st.info("Пожалуйста, загрузите файл для анализа.")
