import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import plotly.express as px

st.set_page_config(page_title="📊 Умный Анализ Данных", layout="wide")
st.title("📊 Интерактивный AI Анализ и Визуализация Данных")
st.markdown("Загрузите CSV, Excel или JSON — и получите подходящие графики под ваши данные.")

# Загрузка данных
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            return pd.DataFrame(data) if isinstance(data, list) else None
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None

# Рекомендации по визуализациям
def recommend_charts(df):
    charts = []
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()

    if len(num_cols) >= 1:
        charts.extend(["Histogram", "Boxplot", "Slope", "Diverging", "Dumbbell"])
    if len(num_cols) >= 2:
        charts.extend(["Scatter", "Bubble", "Line", "Bar Grouped", "Stacked Bubble"])
    if cat_cols and num_cols:
        charts.extend(["Bar", "Horizontal Bar", "Lollipop", "Pie", "Stacked Bar"])
    return list(set(charts))

# Построение графиков
def plot_chart(df, chart_type, x=None, y=None, hue=None, size=None):
    fig = None
    try:
        if chart_type == "Histogram":
            fig = px.histogram(df, x=x, color=hue)
        elif chart_type == "Boxplot":
            fig = px.box(df, x=hue, y=x, points="all")
        elif chart_type == "Bar":
            fig = px.bar(df, x=x, y=y, color=hue)
        elif chart_type == "Horizontal Bar":
            fig = px.bar(df, x=y, y=x, color=hue, orientation='h')
        elif chart_type == "Lollipop":
            fig = px.scatter(df, x=y, y=x, color=hue)
        elif chart_type == "Bar Grouped":
            fig = px.bar(df, x=x, y=y, color=hue, barmode="group")
        elif chart_type == "Stacked Bar":
            fig = px.bar(df, x=x, y=y, color=hue, barmode="stack")
        elif chart_type == "Pie":
            fig = px.pie(df, names=x, values=y)
        elif chart_type == "Line":
            fig = px.line(df, x=x, y=y, color=hue)
        elif chart_type == "Slope":
            fig = px.line(df.sort_values(x), x=x, y=y, color=hue, markers=True)
        elif chart_type == "Diverging":
            df['delta'] = df[y] - df[y].mean()
            fig = px.bar(df, x=x, y='delta', color='delta', color_continuous_scale='RdBu')
        elif chart_type == "Dumbbell":
            fig = px.scatter(df, x=x, y=y, color=hue)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x, y=y, color=hue, size=size)
        elif chart_type == "Bubble":
            fig = px.scatter(df, x=x, y=y, color=hue, size=size, size_max=60)
        elif chart_type == "Stacked Bubble":
            fig = px.scatter(df, x=x, y=y, size=size, color=hue, opacity=0.6)
    except Exception as e:
        st.error(f"Ошибка построения графика: {e}")
    return fig

# Интерфейс
uploaded_file = st.file_uploader("Загрузите файл", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.success(f"✅ Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
        st.dataframe(df.head())

        recommended = recommend_charts(df)
        chart_type = st.selectbox("Выберите тип графика", recommended)

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
        all_cols = df.columns.tolist()

        x = y = hue = size = None

        if chart_type in ["Histogram", "Boxplot"]:
            x = st.selectbox("Числовая колонка", numeric_cols)
            hue = st.selectbox("Группировка (опц.)", [None] + categorical_cols)
        elif chart_type in ["Bar", "Horizontal Bar", "Lollipop", "Bar Grouped", "Stacked Bar"]:
            x = st.selectbox("Категория (X)", categorical_cols)
            y = st.selectbox("Значение (Y)", numeric_cols)
            hue = st.selectbox("Группировка", [None] + categorical_cols)
        elif chart_type in ["Line", "Slope", "Diverging", "Dumbbell"]:
            x = st.selectbox("Ось X (например, время)", all_cols)
            y = st.selectbox("Ось Y", numeric_cols)
            hue = st.selectbox("Группировка", [None] + all_cols)
        elif chart_type == "Pie":
            x = st.selectbox("Категория", categorical_cols)
            y = st.selectbox("Значение", numeric_cols)
        elif chart_type in ["Scatter", "Bubble", "Stacked Bubble"]:
            x = st.selectbox("X", numeric_cols)
            y = st.selectbox("Y", numeric_cols)
            hue = st.selectbox("Цвет (категория)", [None] + all_cols)
            size = st.selectbox("Размер", [None] + numeric_cols)

        if st.button("📈 Построить"):
            fig = plot_chart(df, chart_type, x, y, hue, size)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Ошибка загрузки данных.")
else:
    st.info("⬆ Загрузите данные для начала анализа.")
