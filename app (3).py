import streamlit as st
import pandas as pd
import plotly.express as px
import io
import json
import datetime
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Конфигурация страницы
st.set_page_config(
    page_title="EDA Анализатор",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("📊 EDA Анализатор")
st.markdown("Загрузите файл и получите визуализацию и анализ данных.")

# Загрузка файла пользователем
uploaded_file = st.file_uploader("📁 Загрузите CSV, Excel или JSON файл", type=["csv", "xlsx", "xls", "json"])

# Кэшируем загрузку данных
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

# Отображаем данные
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.subheader("📄 Превью данных")
        st.dataframe(df.head(100))

        with st.expander("🧹 Информация о данных"):
            st.write("**Размерность:**", df.shape)
            st.write("**Типы данных:**")
            st.write(df.dtypes)
            st.write("**Пропущенные значения:**")
            st.write(df.isnull().sum())

        # Профилирование данных
        if st.checkbox("📈 Сгенерировать отчёт Pandas Profiling"):
            profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
            st_profile_report(profile)

        # Визуализация
        st.subheader("📊 Быстрая визуализация")
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        categorical_columns = df.select_dtypes(include='object').columns.tolist()

        plot_type = st.selectbox("Выберите тип графика", ["Гистограмма", "Boxplot", "Точечный график", "Bar chart"])

        if plot_type == "Гистограмма":
            col = st.selectbox("Выберите числовую колонку", numeric_columns)
            fig = px.histogram(df, x=col)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Boxplot":
            col = st.selectbox("Выберите числовую колонку", numeric_columns)
            fig = px.box(df, y=col)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Точечный график":
            x_col = st.selectbox("X ось", numeric_columns)
            y_col = st.selectbox("Y ось", numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
            fig = px.scatter(df, x=x_col, y=y_col, color=df[categorical_columns[0]] if categorical_columns else None)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Bar chart":
            cat_col = st.selectbox("Выберите категориальную колонку", categorical_columns)
            fig = px.bar(df[cat_col].value_counts().reset_index(), x='index', y=cat_col)
            st.plotly_chart(fig, use_container_width=True)
