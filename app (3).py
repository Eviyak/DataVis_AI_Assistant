import streamlit as st
import pandas as pd
import plotly.express as px
import openai

# Загружаем ключ из секрета
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("AI Визуализация данных")

uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Файл загружен!")

    col = st.selectbox("Выберите колонку для визуализации", df.columns)

    def simple_smart_plot(data, column):
        """Автоматический выбор визуализации на основе типа данных"""
        if pd.api.types.is_numeric_dtype(data[column]):
            return px.histogram(data, x=column, title=f'Гистограмма: {column}')
        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            num_cols = data.select_dtypes(include='number').columns
            y_col = num_cols[0] if len(num_cols) > 0 else None
            if y_col:
                return px.line(data, x=column, y=y_col, title=f'Временной ряд: {column} и {y_col}')
        elif data[column].nunique() < 10:
