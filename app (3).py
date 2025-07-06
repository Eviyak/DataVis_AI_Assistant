import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import io
import json
import warnings

from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.api.types import is_datetime64_any_dtype

warnings.filterwarnings("ignore")

# Настройки
st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title("🤖 AI Data Analyzer Pro")

# OpenAI API Key
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# Загрузка данных
@st.cache_data
def load_data(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), encoding_errors="ignore")
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    elif uploaded_file.name.endswith(".json"):
        data = json.loads(file_bytes.decode("utf-8"))
        return pd.json_normalize(data)
    return None

# AI анализ инсайтов и визуализаций
@st.cache_data
def generate_insights_and_visuals(df):
    if not openai.api_key:
        return "❌ API ключ не установлен", [], []

    prompt = (
        f"Ты опытный аналитик данных. На основе следующего DataFrame:\n"
        f"Строк: {df.shape[0]}\n"
        f"Колонки: {list(df.columns)}\n"
        f"Примеры данных:\n{df.head(3).to_dict()}\n\n"
        f"1. Сформулируй инсайты (аналитические выводы).\n"
        f"2. Предложи список визуализаций из этого списка, подходящих под данные: "
        f"[гистограмма, тепловая карта, точечная диаграмма, 3D scatter, временной ряд, candlestick, аномалии, ящик с усами, столбчатая диаграмма, круговая диаграмма, scatter с трендом, pairplot, density plot, violin plot, treemap, area chart, bubble chart, 2D histogram, boxen plot, временной heatmap]\n\n"
        f"Формат JSON:\n"
        f'{{"insights": "...", "visualizations": [{{"viz_type": "...", "x_axis": "...", "y_axis": "...", "z_axis": "...", "color": "...", "size": "..."}}]}}'
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        content = response.choices[0].message.content.strip()

        if not content:
            return "⚠️ GPT вернул пустой ответ", [], None

        try:
            result = json.loads(content)
            return result.get("insights", ""), result.get("visualizations", []), None
        except json.JSONDecodeError as e:
            return f"⚠️ Не удалось разобрать JSON:\n```json\n{content}\n```\n\nОшибка: {e}", [], None

    except Exception as e:
        return f"❌ Ошибка API запроса: {e}", [], None


# Построение графика
def create_visualization(df, viz):
    try:
        viz_type = viz.get("viz_type", "").lower()
        x = viz.get("x_axis")
        y = viz.get("y_axis")
        z = viz.get("z_axis")
        color = viz.get("color")
        size = viz.get("size")

        if viz_type == "гистограмма":
            return px.histogram(df, x=x, color=color)
        elif viz_type == "точечная диаграмма":
            return px.scatter(df, x=x, y=y, color=color, size=size)
        elif viz_type == "3d scatter":
            return px.scatter_3d(df, x=x, y=y, z=z, color=color, size=size)
        elif viz_type == "временной ряд":
            if is_datetime64_any_dtype(df[x]):
                return px.line(df.sort_values(x), x=x, y=y, color=color)
        elif viz_type == "тепловая карта":
            corr = df.select_dtypes(include=np.number).corr()
            return px.imshow(corr, text_auto=True)
        elif viz_type == "ящик с усами":
            return px.box(df, x=x, y=y, color=color)
        elif viz_type == "bubble chart":
            return px.scatter(df, x=x, y=y, size=size, color=color)
        elif viz_type == "violin plot":
            return px.violin(df, x=x, y=y, color=color)
        elif viz_type == "boxen plot":
            return px.box(df, x=x, y=y, color=color, points="all", notched=True)
        elif viz_type == "столбчатая диаграмма":
            return px.bar(df, x=x, y=y, color=color)
        elif viz_type == "круговая диаграмма":
            return px.pie(df, names=x, values=y)
        elif viz_type == "density plot":
            return px.density_contour(df, x=x, y=y)
        elif viz_type == "area chart":
            return px.area(df, x=x, y=y, color=color)
        else:
            return None
    except Exception as e:
        st.warning(f"Ошибка визуализации: {e}")
        return None

# Интерфейс
uploaded_file = st.file_uploader("📁 Загрузите CSV, Excel или JSON файл", type=["csv", "xlsx", "xls", "json"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.subheader("🔍 Просмотр данных")
        st.dataframe(df.head(100), use_container_width=True)

        st.divider()

        # AI-инсайты и визуализации
        with st.spinner("🧠 Генерация аналитики и визуализаций..."):
            insights, visualizations, _ = generate_insights_and_visuals(df)

        # Отображение инсайтов
        st.subheader("🧠 Инсайты")
        st.markdown(insights)

        # Отображение визуализаций
        if visualizations:
            st.subheader("📈 Рекомендуемые визуализации")
            for i, viz in enumerate(visualizations):
                fig = create_visualization(df, viz)
                if fig:
                    st.markdown(f"**{i+1}. {viz['viz_type'].capitalize()}**")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"❌ Невозможно построить визуализацию: {viz}")
        else:
            st.info("AI не предложил визуализации.")
