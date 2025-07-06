import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import io

st.set_page_config(page_title="📊 DataVis AI", layout="wide")
st.title("📊 AI-помощник для анализа данных")
st.markdown("Загрузите CSV, Excel или JSON файл и исследуйте данные через умные графики")

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

uploaded_file = st.file_uploader("Загрузите файл", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.success(f"✅ Загружено {df.shape[0]} строк и {df.shape[1]} колонок")
        st.dataframe(df.head())

        st.subheader("📈 Построение графиков")
        col1, col2 = st.columns(2)

        with col1:
            selected_col1 = st.selectbox("Выберите первую колонку", df.columns)
        with col2:
            selected_col2 = st.selectbox("Выберите вторую колонку (необязательно)", [None] + list(df.columns))

        # Определим типы графиков по данным
        numeric_cols = df.select_dtypes(include='number').columns
        categorical_cols = df.select_dtypes(exclude='number').columns

        available_plots = []

        if selected_col1 and not selected_col2:
            if selected_col1 in numeric_cols:
                available_plots = ['Гистограмма', 'Boxplot', 'Карта плотности (KDE)']
            elif selected_col1 in categorical_cols:
                available_plots = ['Barplot']

        if selected_col1 and selected_col2:
            if selected_col1 in numeric_cols and selected_col2 in numeric_cols:
                available_plots = ['Scatterplot', 'Линейный график']
            elif (selected_col1 in categorical_cols and selected_col2 in numeric_cols) or \
                 (selected_col2 in categorical_cols and selected_col1 in numeric_cols):
                available_plots = ['Boxplot', 'Barplot']

        plot_type = st.selectbox("Тип графика", available_plots if available_plots else ["Недоступно"])

        if st.button("📊 Построить график") and plot_type != "Недоступно":
            fig, ax = plt.subplots(figsize=(8, 5))

            try:
                if plot_type == 'Гистограмма':
                    sns.histplot(df[selected_col1], kde=True, ax=ax)
                elif plot_type == 'Boxplot':
                    if selected_col2:
                        sns.boxplot(x=df[selected_col2], y=df[selected_col1], ax=ax)
                    else:
                        sns.boxplot(y=df[selected_col1], ax=ax)
                elif plot_type == 'Barplot':
                    if selected_col2:
                        sns.barplot(x=df[selected_col1], y=df[selected_col2], ax=ax)
                    else:
                        sns.countplot(x=df[selected_col1], ax=ax)
                elif plot_type == 'Карта плотности (KDE)':
                    sns.kdeplot(df[selected_col1], fill=True, ax=ax)
                elif plot_type == 'Scatterplot':
                    sns.scatterplot(x=df[selected_col1], y=df[selected_col2], ax=ax)
                elif plot_type == 'Линейный график':
                    sns.lineplot(x=df[selected_col1], y=df[selected_col2], ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Ошибка построения графика: {e}")

else:
    st.info("Пожалуйста, загрузите файл для начала анализа.")
