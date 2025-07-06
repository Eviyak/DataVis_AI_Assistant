import streamlit as st
import pandas as pd
import plotly.express as px
import openai

# Устанавливаем API-ключ (в .streamlit/secrets.toml)
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
            counts = data[column].value_counts()
            return px.bar(x=counts.index, y=counts.values, title=f'Бар-чарт: {column}', labels={'x': column, 'y': 'Количество'})
        
        return px.histogram(data, x=column, title=f'Гистограмма (по умолчанию): {column}')

    use_ai = st.checkbox("Использовать GPT-4 для выбора графика", value=False)

    if use_ai:
        with st.spinner("AI анализирует данные..."):
            try:
                description = df[col].describe(include='all').to_string()

                system_prompt = "Ты помощник по визуализации данных. По описанию колонки подскажи лучший тип графика."
                user_prompt = f"""Вот описание столбца '{col}':
{description}

Какой тип графика лучше всего подходит для анализа этого столбца? Ответь одним словом: histogram, bar, line, box или pie."""

                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                suggestion = response.choices[0].message.content.strip().lower()
            except Exception as e:
                st.error(f"Ошибка GPT: {e}")
                suggestion = "histogram"

        st.info(f"GPT предлагает использовать график: **{suggestion}**")

        # Отрисовка графика на основе совета GPT
        if suggestion == "histogram":
            fig = px.histogram(df, x=col, title=f'Гистограмма: {col}')
        elif suggestion == "bar":
            counts = df[col].value_counts()
            fig = px.bar(x=counts.index, y=counts.values, title=f'Бар-чарт: {col}')
        elif suggestion == "line":
            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols) > 0:
                fig = px.line(df, x=col, y=num_cols[0], title=f'Линейный график: {col}')
            else:
                fig = simple_smart_plot(df, col)
        elif suggestion == "box":
            fig = px.box(df, y=col, title=f'Boxplot: {col}')
        elif suggestion == "pie":
            counts = df[col].value_counts()
            fig = px.pie(names=counts.index, values=counts.values, title=f'Круговая диаграмма: {col}')
        else:
            fig = simple_smart_plot(df, col)
    else:
        fig = simple_smart_plot(df, col)

    st.plotly_chart(fig, use_container_width=True)
