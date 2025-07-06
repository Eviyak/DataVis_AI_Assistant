import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_datetime64_any_dtype
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"  # Вставь сюда свой API ключ

# Оптимизация памяти датафрейма
def reduce_mem_usage(df):
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
                df[col] = df[col].astype(np.float32)
    return df

# Загрузка данных из файла
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Неподдерживаемый формат файла.")
            return None
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")
        return None

# Анализ с OpenAI: общие комментарии по данным
@st.cache_data
def analyze_with_ai(df):
    prompt = f"Вот данные:\n{df.head(10).to_dict(orient='records')}\nНапиши краткий анализ этих данных, выяви ключевые особенности."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка при вызове AI: {e}"

# Генерация AI-инсайтов и визуализаций (ожидается JSON с инсайтами и визуализацией)
@st.cache_data
def generate_ai_insights(df):
    prompt = (
        f"Проанализируй данные:\n{df.head(20).to_dict(orient='records')}\n"
        "Верни JSON с ключами:\n"
        "\"insights\" - краткие инсайты,\n"
        "\"viz_type\" - тип визуализации (например, 'точечная диаграмма', 'гистограмма', 'тепловая карта'),\n"
        "\"x_axis\", \"y_axis\", \"z_axis\", \"color\", \"size\" - имена колонок для визуализации или пустые строки если не применимо.\n"
        "Пример вывода:\n"
        '{ "insights": "Инсайты...", "viz_type": "точечная диаграмма", "x_axis": "age", "y_axis": "chol", "z_axis": "", "color": "target", "size": "trestbps" }'
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        json_text = response.choices[0].message.content.strip()
        # Попытка распарсить JSON
        ai_result = json.loads(json_text)
        return ai_result
    except Exception as e:
        return {"insights": f"Ошибка при разборе AI ответа: {e}", "viz_type": "", "x_axis": "", "y_axis": "", "z_axis": "", "color": "", "size": ""}

# Создание визуализации на основе рекомендаций AI
def create_visualization(df, viz_type, x=None, y=None, z=None, color=None, size=None):
    try:
        viz_type = viz_type.lower() if viz_type else ""
        viz_df = df.sample(min(10000, len(df))) if len(df) > 10000 else df

        if viz_type in ['гистограмма', 'histogram', 'hist']:
            if x and x in df.columns:
                fig = px.histogram(viz_df, x=x, color=color if color in df.columns else None)
                return fig

        elif viz_type in ['тепловая карта', 'heatmap']:
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                return fig

        elif viz_type in ['3d scatter', '3d scatter plot', '3d точечная диаграмма']:
            if x and y and z and all(col in df.columns for col in [x, y, z]):
                fig = px.scatter_3d(viz_df, x=x, y=y, z=z,
                                    color=color if color in df.columns else None,
                                    size=size if size in df.columns else None)
                return fig

        elif viz_type in ['временной ряд', 'time series']:
            if x and y and x in df.columns and y in df.columns:
                if is_datetime64_any_dtype(df[x]):
                    fig = px.line(viz_df.sort_values(x), x=x, y=y, color=color if color in df.columns else None)
                    return fig
                else:
                    return None

        elif viz_type in ['точечная диаграмма', 'scatter plot']:
            if x and y and x in df.columns and y in df.columns:
                fig = px.scatter(viz_df, x=x, y=y,
                                 color=color if color in df.columns else None,
                                 size=size if size in df.columns else None)
                return fig

        # Если ни один вариант не подошел
        return None
    except Exception as e:
        st.error(f"Ошибка построения визуализации: {str(e)}")
        return None

# Временной ряд для первого датасета с датой и числом
def time_series_analysis(df, date_col, value_col):
    try:
        if date_col in df.columns and value_col in df.columns:
            if not is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            fig = px.line(df.sort_values(date_col), x=date_col, y=value_col)
            return fig
    except Exception:
        return None

def main():
    st.title("AI Data Insights & Visualization Assistant")
    uploaded_file = st.file_uploader("Загрузите файл данных (CSV, Excel, JSON)", type=['csv', 'xlsx', 'xls', 'json'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            df = reduce_mem_usage(df)
            st.subheader("Просмотр данных")
            st.dataframe(df.head(1000))

            st.subheader("Обзор данных с помощью AI")
            analysis_text = analyze_with_ai(df)
            st.markdown(analysis_text)

            st.subheader("AI Инсайты и рекомендации")
            ai_result = generate_ai_insights(df)

            st.markdown(f"**Инсайты:**\n\n{ai_result.get('insights', 'Нет инсайтов')}")

            viz_fig = create_visualization(
                df,
                viz_type=ai_result.get('viz_type'),
                x=ai_result.get('x_axis'),
                y=ai_result.get('y_axis'),
                z=ai_result.get('z_axis'),
                color=ai_result.get('color'),
                size=ai_result.get('size'),
            )

            if viz_fig:
                st.plotly_chart(viz_fig, use_container_width=True)
            else:
                st.info("AI не предложил подходящую визуализацию для ваших данных.")

            # Показать временной ряд, если есть дата и числовой столбец
            date_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
            num_cols = df.select_dtypes(include=np.number).columns

            if len(date_cols) > 0 and len(num_cols) > 0:
                st.subheader("Временной ряд")
                date_col = date_cols[0]
                value_col = num_cols[0]
                ts_fig = time_series_analysis(df, date_col, value_col)
                if ts_fig:
                    st.plotly_chart(ts_fig, use_container_width=True)

if __name__ == "__main__":
    main()
