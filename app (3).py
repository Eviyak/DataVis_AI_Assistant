import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import openai
import io
import json
from sklearn.ensemble import IsolationForest

# ============================ Настройки ============================
st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("🤖 AI Data Analyst Assistant")
st.markdown("Загрузите файл и получите автоматические инсайты и визуализации от AI.")

# ============================ Настройки OpenAI ============================
openai.api_key = st.secrets.get("OPENAI_API_KEY", "sk-...")  # Замените своим ключом, если не используете secrets

# ============================ Функции ============================

@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            return pd.read_json(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        return None

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type == object or col_type.name == 'category':
            continue
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Память уменьшена с {start_mem:.2f}MB до {end_mem:.2f}MB")
    return df

def get_openai_response(prompt, model="gpt-4", max_tokens=500):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "Ты — эксперт по анализу данных."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message["content"]
    except Exception as e:
        st.error(f"Ошибка OpenAI: {e}")
        return None

def generate_ai_insights_and_viz(df):
    sample_data = df.sample(min(100, len(df))).to_dict(orient="records")
    prompt = f"""
Ты — аналитик данных. Проанализируй предоставленные данные и:
1. Выведи ключевые инсайты (макс. 4 пункта).
2. Предложи один лучший тип графика (пример: bar, line, scatter, pie, histogram, box, heatmap, 3d_scatter).
3. Укажи, какие переменные использовать (x, y, [z], [color], [size]).

Формат ответа в JSON:
{{
  "insights": "…",
  "viz_type": "scatter",
  "x": "feature1",
  "y": "feature2",
  "z": null,
  "color": "feature3",
  "size": null
}}

Вот данные:
{json.dumps(sample_data, ensure_ascii=False)[:4000]}  # ограничиваем размер
"""
    response = get_openai_response(prompt)
    try:
        parsed = json.loads(response)
        return parsed.get("insights"), parsed.get("viz_type"), parsed.get("x"), parsed.get("y"), parsed.get("z"), parsed.get("color"), parsed.get("size")
    except:
        st.warning("Не удалось распарсить AI-ответ.")
        return None, None, None, None, None, None, None

def create_visualization(df, chart_type, x, y, z=None, color=None, size=None):
    try:
        if chart_type == "scatter":
            return px.scatter(df, x=x, y=y, color=color, size=size)
        elif chart_type == "line":
            return px.line(df, x=x, y=y, color=color)
        elif chart_type == "bar":
            return px.bar(df, x=x, y=y, color=color)
        elif chart_type == "box":
            return px.box(df, x=x, y=y, color=color)
        elif chart_type == "histogram":
            return px.histogram(df, x=x, color=color)
        elif chart_type == "pie":
            return px.pie(df, names=x, values=y)
        elif chart_type == "heatmap":
            corr = df.select_dtypes(include=np.number).corr()
            return px.imshow(corr, text_auto=True)
        elif chart_type == "3d_scatter":
            return px.scatter_3d(df, x=x, y=y, z=z, color=color, size=size)
    except Exception as e:
        st.warning(f"Ошибка при создании графика: {e}")
        return None

def analyze_with_ai(df):
    try:
        sample = df.head(100).to_dict(orient="records")
        prompt = f"""Предоставлены данные. Выполни:
- Проверку на пропущенные значения
- Анализ распределения признаков
- Найди возможные закономерности и выбросы

Ответ дай в виде краткого текста, понятного бизнесу.

Данные:
{json.dumps(sample, ensure_ascii=False)[:4000]}
"""
        return get_openai_response(prompt, max_tokens=700)
    except:
        return "Ошибка анализа."

def detect_anomalies(df, column):
    try:
        model = IsolationForest(contamination=0.01)
        df = df[[column]].dropna()
        df["anomaly"] = model.fit_predict(df[[column]])
        return df[df["anomaly"] == -1]
    except Exception as e:
        st.warning(f"Ошибка при поиске аномалий: {e}")
        return None

def time_series_analysis(df, date_col, value_col):
    try:
        df = df[[date_col, value_col]].dropna()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        return px.line(df, x=date_col, y=value_col, title="Анализ временного ряда")
    except Exception as e:
        st.warning(f"Ошибка временного ряда: {e}")
        return None

# ============================ UI ============================

uploaded_file = st.file_uploader("Загрузите файл с данными (.csv, .xlsx, .json)", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        df = reduce_mem_usage(df)

        st.subheader("🔍 Предварительный просмотр данных")
        st.dataframe(df.head(100))

        insights, viz_type, x_axis, y_axis, z_axis, color, size = generate_ai_insights_and_viz(df)

        st.subheader("🤖 AI-инсайты")
        if insights:
            st.markdown(insights)
        else:
            st.info("AI не смог сгенерировать инсайты.")

        st.subheader("📊 Рекомендованная визуализация от AI")
        if viz_type:
            st.markdown(f"**Тип графика:** `{viz_type}`")
            fig = create_visualization(df, viz_type, x_axis, y_axis, z_axis, color, size)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("❌ Не удалось построить визуализацию.")
        else:
            st.info("AI не предложил визуализацию.")

        with st.expander("📎 Дополнительный анализ"):
            st.markdown(analyze_with_ai(df))

            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                col_for_anom = st.selectbox("Выберите числовой столбец для поиска аномалий", num_cols)
                anomalies = detect_anomalies(df, col_for_anom)
                if anomalies is not None and not anomalies.empty:
                    st.success(f"Найдено {len(anomalies)} аномалий в столбце `{col_for_anom}`")
                    st.dataframe(anomalies.head(20))

                    fig_anom = px.scatter(df, x=df.index, y=col_for_anom, title=f"Аномалии в {col_for_anom}")
                    fig_anom.add_trace(go.Scatter(
                        x=anomalies.index, y=anomalies[col_for_anom],
                        mode='markers', marker=dict(color='red', size=10),
                        name='Аномалии'
                    ))
                    st.plotly_chart(fig_anom, use_container_width=True)

            date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
            if not date_cols:
                for c in df.columns:
                    try:
                        df[c] = pd.to_datetime(df[c])
                        date_cols.append(c)
                    except:
                        continue

            if date_cols and num_cols:
                date_col = st.selectbox("Выберите столбец с датами", date_cols)
                value_col = st.selectbox("Выберите числовой столбец для временного ряда", num_cols)
                fig_ts = time_series_analysis(df, date_col, value_col)
                if fig_ts:
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.warning("⚠️ Не удалось построить временной ряд.")
else:
    st.info("⬆️ Загрузите файл для анализа.")
