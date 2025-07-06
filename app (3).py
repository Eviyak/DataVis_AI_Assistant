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

# Настройки страницы
st.set_page_config(
    page_title="🤖 AI Data Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка API ключа OpenAI из Streamlit Secrets
if 'OPENAI_API_KEY' in st.secrets:
    openai.api_key = st.secrets['OPENAI_API_KEY']
else:
    openai.api_key = ""

# Тема дневная (без выбора)
st.markdown("""
    <style>
        .stApp { background-color: #f0f2f6; color: #000000; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# Заголовок
st.title("🤖 AI Data Analyzer Pro")
st.markdown("""
    <div style="background-color:#ffffff;padding:10px;border-radius:10px;margin-bottom:20px;">
    <p style="color:#333;font-size:18px;">🚀 <b>Автоматический анализ данных с AI-powered инсайтами</b></p>
    <p style="color:#666;">Загрузите CSV, Excel или JSON — получите полный анализ и визуализацию</p>
    </div>
""", unsafe_allow_html=True)

# Кэшированная функция загрузки данных
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

# Оптимизация памяти
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

# AI анализ данных (описание и статистика)
@st.cache_data(show_spinner="Анализирую данные... 🔍", ttl=600)
def analyze_with_ai(df):
    try:
        analysis = f"### 📊 Общий обзор данных\n"
        analysis += f"- **Строки:** {df.shape[0]}\n"
        analysis += f"- **Колонки:** {df.shape[1]}\n"
        analysis += f"- **Объем данных:** {df.memory_usage().sum() / 1024**2:.2f} MB\n\n"

        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            analysis += "### 🔢 Числовые данные\n"
            stats = df[num_cols].describe().transpose()
            stats['skew'] = df[num_cols].skew()
            analysis += stats[['mean', 'std', 'min', '50%', 'max', 'skew']].to_markdown()

        cat_cols = df.select_dtypes(exclude=np.number).columns
        if len(cat_cols) > 0:
            analysis += "\n\n### 🔤 Категориальные данные\n"
            for col in cat_cols:
                analysis += f"- **{col}**: {df[col].nunique()} уникальных значений\n"

        missing = df.isnull().sum()
        if missing.sum() > 0:
            analysis += "\n\n### ⚠️ Пропущенные значения\n"
            missing_percent = missing[missing > 0] / len(df) * 100
            missing_df = pd.DataFrame({'Колонка': missing_percent.index,
                                      'Пропуски': missing[missing > 0],
                                      '%': missing_percent.values.round(1)})
            analysis += missing_df.to_markdown(index=False)

        if len(num_cols) > 1:
            corr = df[num_cols].corr().abs().unstack().sort_values(ascending=False)
            strong_corr = corr[(corr > 0.7) & (corr < 1)].drop_duplicates()
            if len(strong_corr) > 0:
                analysis += "\n\n### 🔗 Сильные корреляции\n"
                for pair, value in strong_corr.items():
                    analysis += f"- {pair[0]} и {pair[1]}: {value:.2f}\n"

        return analysis
    except Exception as e:
        return f"Ошибка анализа: {str(e)}"

# Обнаружение аномалий IsolationForest
@st.cache_data(show_spinner="Ищу аномалии... 🕵️", ttl=300)
def detect_anomalies(df, column):
    try:
        if len(df) > 10000:
            sample = df.sample(min(5000, len(df)))
        else:
            sample = df

        model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        model.fit(sample[[column]])
        df['anomaly'] = model.predict(df[[column]])
        anomalies = df[df['anomaly'] == -1]
        return anomalies
    except:
        return None

# Анализ временных рядов с декомпозицией
@st.cache_data(show_spinner="Анализирую временные ряды... ⏳", ttl=300)
def time_series_analysis(df, date_col, value_col):
    try:
        df = df.set_index(date_col).sort_index()
        if len(df) > 1000:
            df = df.resample('D').mean()

        decomposition = seasonal_decompose(df[value_col], period=min(12, len(df)//2))

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=df.index, y=df[value_col], name='Исходные данные'), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Тренд'), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Сезонность'), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Остатки'), row=4, col=1)

        fig.update_layout(height=800, title_text="Декомпозиция временного ряда")
        return fig
    except:
        return None

# Генерация инсайтов и рекомендации по визуализации через OpenAI GPT
@st.cache_data(show_spinner="Генерирую AI инсайты и рекомендации... 🤖", ttl=600)
def generate_ai_insights_and_viz(df):
    if not openai.api_key:
        return "🔑 Ключ OpenAI API не установлен. Добавьте его в Secrets.", None, None, None, None, None, None

    prompt = (
    f"Ты аналитик данных. Сделай краткий аналитический отчет и дай рекомендации по данным.\n"
    f"Данные: {df.shape[0]} строк, {df.shape[1]} колонок.\n"
    f"Колонки: {list(df.columns)}.\n"
    f"Первые 5 строк:\n{df.head().to_dict()}\n\n"
    f"Напиши инсайты и предложи несколько типов визуализаций из списка:\n"
    f"- гистограмма\n"
    f"- тепловая карта\n"
    f"- точечная диаграмма\n"
    f"- 3D scatter\n"
    f"- временной ряд\n"
    f"- candlestick\n"
    f"- аномалии\n"
    f"- ящик с усами\n"
    f"- столбчатая диаграмма\n"
    f"- круговая диаграмма\n"
    f"- scatter с трендом\n"
    f"- pairplot\n"
    f"- density plot\n"
    f"- violin plot\n"
    f"- treemap\n"
    f"- area chart\n"
    f"- bubble chart\n"
    f"- 2D histogram\n"
    f"- boxen plot\n"
    f"- временной heatmap\n\n"
    f"Ответь в JSON формате:\n"
    f'{{"insights": "...", "visualizations": [{{"viz_type": "...", "x_axis": "...", "y_axis": "...", "z_axis": "...", "color": "...", "size": "..."}}]}}\n'
    f"Если какой-то тип визуализации не подходит, просто пропусти его."
)

    )

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
        text = response['choices'][0]['message']['content']

        try:
            parsed = json.loads(text)
            insights = parsed.get('insights', '')
            viz_type = parsed.get('viz_type', None)
            x_axis = parsed.get('x_axis', None)
            y_axis = parsed.get('y_axis', None)
            z_axis = parsed.get('z_axis', None)
            color = parsed.get('color', None)
            size = parsed.get('size', None)
            return insights, viz_type, x_axis, y_axis, z_axis, color, size

        except json.JSONDecodeError:
            return text, None, None, None, None, None, None

    except Exception as e:
        return f"Ошибка вызова OpenAI API: {str(e)}", None, None, None, None, None, None

# Визуализация данных
def create_visualization(df, viz_type, x=None, y=None, z=None, color=None, size=None):
    try:
        viz_df = df.sample(min(10000, len(df))) if len(df) > 10000 else df

        if viz_type == "Гистограмма":
            fig = px.histogram(viz_df, x=x, color=color, marginal="box", nbins=50)

        elif viz_type == "Тепловая карта":
            corr = viz_df.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Корреляционная матрица")

        elif viz_type == "3D scatter":
            fig = px.scatter_3d(viz_df, x=x, y=y, z=z, color=color, size=size)

        elif viz_type == "Временной ряд":
            if is_datetime64_any_dtype(viz_df[x]):
                fig = px.line(viz_df.sort_values(x), x=x, y=y, color=color)
            else:
                return None

        elif viz_type == "candlestick":
            # Проверяем нужные колонки
            required = {'open', 'high', 'low', 'close'}
            if required.issubset(set(viz_df.columns)):
                fig = go.Figure(data=[go.Candlestick(
                    x=viz_df[x] if x else viz_df.index,
                    open=viz_df['open'],
                    high=viz_df['high'],
                    low=viz_df['low'],
                    close=viz_df['close']
                )])
            else:
                return None

        elif viz_type == "аномалии":
            anomalies = detect_anomalies(viz_df, y)
            if anomalies is None or anomalies.empty:
                return None
            fig = px.scatter(viz_df, x=x, y=y, title="Аномалии в данных")
            fig.add_trace(go.Scatter(x=anomalies[x], y=anomalies[y], mode='markers',
                                     marker=dict(color='red', size=10), name='Аномалии'))

        elif viz_type == "точечная диаграмма":
            fig = px.scatter(viz_df, x=x, y=y, color=color, size=size)

        else:
            return None

        fig.update_layout(height=600, margin=dict(t=50, b=50, l=50, r=50))
        return fig

    except Exception as e:
        st.warning(f"Ошибка визуализации: {str(e)}")
        return None

# --- Основной интерфейс ---

uploaded_file = st.file_uploader("Загрузите файл с данными (.csv, .xlsx, .json)", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        df = reduce_mem_usage(df)
        st.subheader("Предварительный просмотр данных")
        st.dataframe(df.head(100))

        # AI анализ данных и визуализация
        insights, viz_type, x_axis, y_axis, z_axis, color, size = generate_ai_insights_and_viz(df)

        st.subheader("🤖 AI инсайты")
        st.markdown(insights if insights else "Нет данных для инсайтов.")

        if viz_type:
            st.subheader(f"📈 Рекомендованная визуализация: {viz_type}")
            fig = create_visualization(df, viz_type, x_axis, y_axis, z_axis, color, size)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Не удалось построить визуализацию с предложенными параметрами.")

        else:
            st.info("AI не предложил подходящую визуализацию для ваших данных.")

        # Расширенный анализ
        with st.expander("Дополнительный анализ"):
            st.markdown(analyze_with_ai(df))

            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                col_for_anom = st.selectbox("Выберите числовой столбец для поиска аномалий", num_cols)
                anomalies = detect_anomalies(df, col_for_anom)
                if anomalies is not None and not anomalies.empty:
                    st.write(f"Найдено {len(anomalies)} аномалий в столбце {col_for_anom}")
                    st.dataframe(anomalies.head(20))
                    fig_anom = px.scatter(df, x=df.index, y=col_for_anom, title=f"Аномалии в {col_for_anom}")
                    fig_anom.add_trace(go.Scatter(x=anomalies.index, y=anomalies[col_for_anom],
                                                  mode='markers', marker=dict(color='red', size=10),
                                                  name='Аномалии'))
                    st.plotly_chart(fig_anom, use_container_width=True)

            date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
            if not date_cols:
                # Попытка конвертировать некоторые колонки
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
                    st.info("Не удалось выполнить декомпозицию временного ряда.")

else:
    st.info("Пожалуйста, загрузите файл с данными для анализа.")
