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

# --- Настройки страницы ---
st.set_page_config(
    page_title="🤖 AI Data Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'OPENAI_API_KEY' in st.secrets:
    openai.api_key = st.secrets['OPENAI_API_KEY']
else:
    openai.api_key = ""

st.markdown("""
    <style>
        .stApp { background-color: #f0f2f6; color: #000000; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

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

def get_openai_response(prompt):
    if not openai.api_key:
        return "🔑 Ключ OpenAI API не установлен. Добавьте его в Secrets."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты аналитик данных и визуализации."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Ошибка вызова OpenAI API: {str(e)}"

@st.cache_data(show_spinner="Генерирую AI инсайты и рекомендации... 🤖", ttl=600)
def generate_ai_insights_and_viz_rus(df):
    try:
        sample_data = df.sample(min(100, len(df))).to_dict(orient="records")
    except Exception:
        sample_data = df.head(5).to_dict(orient="records")

    prompt = f"""
Ты — аналитик данных. Проанализируй данные и дай ответ в таком формате (обязательно, на русском языке):
{{
  "insights": "текст инсайтов на русском",
  "viz_type": "название типа графика на русском (например, точечная диаграмма, гистограмма, линейный график)",
  "x_axis": "название колонки для оси X",
  "y_axis": "название колонки для оси Y",
  "z_axis": "название колонки для оси Z или пустая строка",
  "color": "название колонки для цвета или пустая строка",
  "size": "название колонки для размера или пустая строка"
}}

Данные (пример первых 100 строк):
{json.dumps(sample_data, ensure_ascii=False)[:4000]}
"""
    response = get_openai_response(prompt)
    try:
        return json.loads(response)
    except Exception as e:
        st.warning(f"Не удалось распарсить AI-ответ: {e}")
        return {"insights": response, "viz_type": "", "x_axis": "", "y_axis": "", "z_axis": "", "color": "", "size": ""}

def display_ai_insights_rus(ai_response):
    st.markdown("### 🤖 AI Инсайты")
    st.markdown(ai_response.get('insights', 'Нет данных для инсайтов.'))

def display_ai_viz_recommendations(ai_response):
    st.markdown("### 📊 Рекомендации по визуализации")
    st.markdown(f"- Тип графика: **{ai_response.get('viz_type', '')}**")
    st.markdown(f"- Ось X: **{ai_response.get('x_axis', '')}**")
    st.markdown(f"- Ось Y: **{ai_response.get('y_axis', '')}**")

    z = ai_response.get('z_axis', '')
    if z:
        st.markdown(f"- Ось Z: **{z}**")
    color = ai_response.get('color', '')
    if color:
        st.markdown(f"- Цвет: **{color}**")
    size = ai_response.get('size', '')
    if size:
        st.markdown(f"- Размер: **{size}**")

def create_visualization(df, viz_type, x=None, y=None, z=None, color=None, size=None):
    try:
        viz_df = df.sample(min(10000, len(df))) if len(df) > 10000 else df

        # Соответствие русских названий типам plotly
        mapping = {
            "гистограмма": "histogram",
            "тепловая карта": "heatmap",
            "3d scatter": "scatter_3d",
            "3d точечная диаграмма": "scatter_3d",
            "точечная диаграмма": "scatter",
            "линейный график": "line",
            "линейный ряд": "line",
            "candlestick": "candlestick",
            "аномалии": "anomalies"
        }

        viz_key = viz_type.lower()
        if viz_key not in mapping:
            return None

        if mapping[viz_key] == "histogram":
            fig = px.histogram(viz_df, x=x, color=color, marginal="box", nbins=50)

        elif mapping[viz_key] == "heatmap":
            corr = viz_df.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Корреляционная матрица")

        elif mapping[viz_key] == "scatter_3d":
            fig = px.scatter_3d(viz_df, x=x, y=y, z=z, color=color, size=size)

        elif mapping[viz_key] == "scatter":
            fig = px.scatter(viz_df, x=x, y=y, color=color, size=size)

        elif mapping[viz_key] == "line":
            if x and y and is_datetime64_any_dtype(viz_df[x]):
                fig = px.line(viz_df.sort_values(x), x=x, y=y, color=color)
            else:
                fig = px.line(viz_df, x=x, y=y, color=color)

        elif mapping[viz_key] == "candlestick":
            fig = go.Figure(data=[go.Candlestick(x=viz_df[x],
                                                open=viz_df['open'], high=viz_df['high'],
                                                low=viz_df['low'], close=viz_df['close'])])
            fig.update_layout(title="Свечной график")

        else:
            fig = None

        if fig:
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), template="plotly_white")
        return fig
    except Exception as e:
        st.warning(f"Не удалось построить график: {e}")
        return None

def main():
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV, Excel или JSON", type=["csv", "xlsx", "xls", "json"])

    if uploaded_file is None:
        st.info("Пожалуйста, загрузите файл с данными для анализа.")
        return

    df = load_data(uploaded_file)
    if df is None or df.empty:
        st.error("Не удалось загрузить данные или файл пуст.")
        return

    df = reduce_mem_usage(df)

    st.sidebar.markdown("### Статистика данных")
    st.sidebar.write(df.describe(include='all').T)

    # --- Запускаем AI ---
    with st.spinner("Генерирую AI инсайты и рекомендации..."):
        ai_response = generate_ai_insights_and_viz_rus(df)

    # --- Отображаем инсайты ---
    display_ai_insights_rus(ai_response)

    st.markdown("---")

    # --- Отображаем рекомендации по визуализации ---
    display_ai_viz_recommendations(ai_response)

    # --- Строим визуализацию ---
    viz_type = ai_response.get('viz_type', '').lower()
    x = ai_response.get('x_axis', '')
    y = ai_response.get('y_axis', '')
    z = ai_response.get('z_axis', '')
    color = ai_response.get('color', '')
    size = ai_response.get('size', '')

    fig = create_visualization(df, viz_type, x, y, z, color, size)

    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("AI не рекомендовал визуализацию или тип графика не поддерживается.")

if __name__ == "__main__":
    main()
