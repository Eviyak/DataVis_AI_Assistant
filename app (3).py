import streamlit as st
import pandas as pd
import plotly.express as px
import openai

# Подключение OpenAI API
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("AI Визуализация и анализ данных")

uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Файл успешно загружен!")

    st.subheader("Общие сведения о данных")
    st.write("Форма данных:", df.shape)
    st.write("Типы данных:")
    st.write(df.dtypes)
    st.write("Пример данных:")
    st.dataframe(df.head())

    col = st.selectbox("Выберите колонку для визуализации", df.columns)

    st.subheader("Статистическое описание колонки")
    try:
        st.text(df[col].describe(include='all').to_string())
    except Exception as e:
        st.warning(f"Ошибка анализа: {e}")

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

    use_ai = st.checkbox("Использовать GPT-4 для выбора графика и инсайтов", value=True)

    if use_ai:
        with st.spinner("AI анализирует данные..."):
            try:
                description = df[col].describe(include='all').to_string()

                system_prompt = "Ты эксперт по визуализации и анализу данных. По описанию столбца предложи лучший тип графика и сделай краткий анализ."
                user_prompt = f"""Вот описание столбца '{col}':
{description}

1. Какой тип графика лучше всего подходит для анализа этого столбца? Ответь ОДНИМ словом (в нижнем регистре): histogram, bar, line, box или pie.
2. Затем, кратко напиши инсайт о данных в этом столбце (на русском языке).
"""

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                answer = response.choices[0].message.content.strip()
                lines = answer.splitlines()
                suggestion = lines[0].strip().lower()
                insight = "\n".join(lines[1:]).strip()

                st.info(f"💡 **GPT предложил тип графика:** `{suggestion}`")
                if insight:
                    st.success(f"📊 **AI-инсайт:**\n\n{insight}")

            except Exception as e:
                st.error(f"Ошибка GPT: {e}")
                suggestion = "histogram"

        # Построение графика по совету GPT
        if suggestion == "histogram":
            fig = px.histogram(df, x=col, title=f'Гистограмма: {col}')
        elif suggestion == "bar":
            counts = df[col].value_counts()
            fig = px.bar(x=counts.index, y=counts.values, title=f'Бар-чарт: {col}', labels={'x': col, 'y': 'Количество'})
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
