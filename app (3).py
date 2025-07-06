import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import openai
import json
import datetime
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import warnings
import io
import time
warnings.filterwarnings('ignore')

# Настройки страницы для Streamlit Cloud
st.set_page_config(
    page_title="🤖 AI Data Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка переменных окружения из Streamlit Secrets
if 'OPENAI_API_KEY' in st.secrets:
    openai.api_key = st.secrets['OPENAI_API_KEY']
else:
    openai.api_key = ""

# Темная/светлая тема
theme = st.sidebar.radio("🎨 Тема", ["Светлая", "Темная"])
if theme == "Темная":
    plt.style.use('dark_background')
    st.markdown("""
        <style>
            .stApp { background-color: #1E1E1E; }
            .st-bb { background-color: transparent; }
            .st-at { background-color: #2E2E2E; }
            .css-1d391kg { color: white; }
            footer { visibility: hidden; }
        </style>
    """, unsafe_allow_html=True)

# Заголовок
st.title("🤖 AI Data Analyzer Pro")
st.markdown("""
    <div style="background-color:#f0f2f6;padding:10px;border-radius:10px;margin-bottom:20px;">
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

# Функция для уменьшения объема данных
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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
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

# AI анализ данных
@st.cache_data(show_spinner="Анализирую данные... 🔍", ttl=600)
def analyze_with_ai(df):
    try:
        analysis = f"### 📊 Общий обзор данных\n"
        analysis += f"- **Строки:** {df.shape[0]}\n"
        analysis += f"- **Колонки:** {df.shape[1]}\n"
        analysis += f"- **Объем данных:** {df.memory_usage().sum() / 1024**2:.2f} MB\n\n"
        
        # Числовые колонки
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            analysis += "### 🔢 Числовые данные\n"
            stats = df[num_cols].describe().transpose()
            stats['skew'] = df[num_cols].skew()
            analysis += stats[['mean', 'std', 'min', '50%', 'max', 'skew']].to_markdown()
        
        # Категориальные колонки
        cat_cols = df.select_dtypes(exclude=np.number).columns
        if len(cat_cols) > 0:
            analysis += "\n\n### 🔤 Категориальные данные\n"
            for col in cat_cols:
                analysis += f"- **{col}**: {df[col].nunique()} уникальных значений\n"
        
        # Пропущенные значения
        missing = df.isnull().sum()
        if missing.sum() > 0:
            analysis += "\n\n### ⚠️ Пропущенные значения\n"
            missing_percent = missing[missing > 0] / len(df) * 100
            missing_df = pd.DataFrame({'Колонка': missing_percent.index, 
                                      'Пропуски': missing[missing > 0], 
                                      '%': missing_percent.values.round(1)})
            analysis += missing_df.to_markdown(index=False)
        
        # Корреляции
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

# Обнаружение аномалий
@st.cache_data(show_spinner="Ищу аномалии... 🕵️", ttl=300)
def detect_anomalies(df, column):
    try:
        if len(df) > 10000:  # Для больших данных используем выборку
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

# Временной анализ
@st.cache_data(show_spinner="Анализирую временные ряды... ⏳", ttl=300)
def time_series_analysis(df, date_col, value_col):
    try:
        df = df.set_index(date_col).sort_index()
        if len(df) > 1000:
            df = df.resample('D').mean()  # Ресемплинг для больших данных
        
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

# Генерация инсайтов с GPT
@st.cache_data(show_spinner="Генерирую AI инсайты... 🤖", ttl=600)
def generate_ai_insights(df):
    try:
        # Если ключ API не установлен, используем локальную логику
        if not openai.api_key:
            return "🔑 Ключ OpenAI API не установлен. Добавьте его в Secrets для расширенного анализа."
            
        # Создаем компактное описание данных
        data_summary = f"Данные содержат {len(df)} строк и {len(df.columns)} колонок:\n"
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique = len(df[col].unique())
            nulls = df[col].isnull().sum()
            data_summary += f"- {col}: {dtype}, {unique} уникальных, {nulls} пропусков\n"
            
        # Формируем промпт
        prompt = f"""
        Ты профессиональный аналитик данных. Проанализируй этот датасет:
        {data_summary}
        
        Сделай 3-5 ключевых вывода на русском языке. Будь конкретным. 
        Если есть числовые колонки, укажи возможные корреляции. 
        Если есть временные ряды, предложи методы анализа.
        Формат: краткие пункты с emoji.
        """
        
        # Вызов OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты опытный data scientist, специализирующийся на анализе данных."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка генерации инсайтов: {str(e)}"

# Визуализации
def create_visualization(df, viz_type, x=None, y=None, z=None, color=None, size=None):
    try:
        # Прогресс-бар для больших данных
        progress = st.progress(0)
        progress.progress(20)
        
        # Ограничение размера данных для визуализации
        viz_df = df.sample(min(10000, len(df))) if len(df) > 10000 else df
        
        fig = None
        if viz_type == "Гистограмма":
            fig = px.histogram(viz_df, x=x, color=color, marginal="box", nbins=50)
        
        elif viz_type == "Тепловая карта":
            corr = viz_df.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')
        
        elif viz_type == "3D Scatter":
            fig = px.scatter_3d(viz_df, x=x, y=y, z=z, color=color, size=size)
        
        elif viz_type == "Аномалии":
            anomalies = detect_anomalies(df, x)
            fig = px.scatter(viz_df, x=x, y=y, color=df.index.isin(anomalies.index))
        
        elif viz_type == "Временной ряд":
            if len(viz_df) > 1000:
                viz_df = viz_df.set_index(x).resample('D').mean().reset_index()
            fig = px.line(viz_df, x=x, y=y, color=color)
            if len(viz_df) > 30:
                viz_df['rolling'] = viz_df[y].rolling(7).mean()
                fig.add_scatter(x=viz_df[x], y=viz_df['rolling'], name='Скользящее среднее (7)')
        
        elif viz_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=viz_df[x],
                open=viz_df[y],
                high=viz_df[y]+viz_df[y].std(),
                low=viz_df[y]-viz_df[y].std(),
                close=viz_df[y])])
        
        else:
            fig = px.scatter(viz_df, x=x, y=y, color=color, size=size)
        
        progress.progress(80)
        
        if fig:
            fig.update_layout(
                template="plotly_dark" if theme == "Темная" else "plotly_white",
                hovermode="x unified",
                height=600
            )
            
            # Оптимизация для больших данных
            fig.update_traces(marker=dict(size=5, opacity=0.7))
        
        progress.progress(100)
        time.sleep(0.2)
        progress.empty()
        return fig
        
    except Exception as e:
        st.error(f"Ошибка визуализации: {str(e)}")
        return None

# Основной интерфейс
with st.sidebar:
    st.header("📤 Загрузка данных")
    uploaded_file = st.file_uploader("Выберите файл", type=["csv", "xlsx", "json"], label_visibility="collapsed")
    
    if uploaded_file:
        st.info(f"Файл: {uploaded_file.name}")
        st.caption("Поддерживаемые форматы: CSV, Excel, JSON")

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        # Оптимизация памяти
        df = reduce_mem_usage(df)
        
        # Основные вкладки
        tab1, tab2, tab3 = st.tabs(["🔍 Обзор данных", "📊 Визуализация", "🤖 AI Анализ"])
        
        with tab1:
            st.subheader("Превью данных")
            st.dataframe(df.head(), use_container_width=True)
            
            with st.expander("🔧 Техническая информация", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Всего строк", len(df))
                    st.metric("Всего колонок", len(df.columns))
                    
                    # Пропущенные значения
                    missing = df.isnull().sum().sum()
                    st.metric("Пропущенные значения", f"{missing} ({missing/df.size:.1%})")
                
                with c2:
                    st.metric("Объем памяти", f"{df.memory_usage().sum() / 1024**2:.2f} MB")
                    
                    # Типы данных
                    dtypes = df.dtypes.value_counts()
                    for dtype, count in dtypes.items():
                        st.caption(f"{dtype}: {count} колонок")
            
            with st.expander("📈 Быстрая визуализация", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    x_quick = st.selectbox("Ось X", df.columns, key="x_quick")
                with col2:
                    y_quick = st.selectbox("Ось Y", [None] + df.select_dtypes(include=np.number).columns.tolist(), key="y_quick")
                
                if st.button("Быстрый график", use_container_width=True):
                    if y_quick:
                        fig = px.scatter(df, x=x_quick, y=y_quick, hover_data=df.columns)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.histogram(df, x=x_quick)
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Расширенная визуализация")
            
            cols = st.columns([1, 3])
            with cols[0]:
                viz_type = st.selectbox("Тип визуализации", [
                    "Гистограмма", "Тепловая карта", "3D Scatter", 
                    "Аномалии", "Временной ряд", "Candlestick"
                ], key="viz_type")
                
                st.divider()
                
                x_axis = st.selectbox("Ось X", df.columns, index=0, key="x_axis")
                
                if viz_type not in ["Тепловая карта"]:
                    y_options = [col for col in df.columns if col != x_axis]
                    y_axis = st.selectbox("Ось Y", y_options, index=min(1, len(y_options)-1), key="y_axis")
                
                if viz_type == "3D Scatter":
                    z_options = [col for col in df.select_dtypes(include=np.number).columns if col not in [x_axis, y_axis]]
                    z_axis = st.selectbox("Ось Z", z_options, key="z_axis")
                else:
                    z_axis = None
                
                color = st.selectbox("Цвет", [None] + [col for col in df.columns if col not in [x_axis, y_axis]], key="color")
                
                if viz_type in ["Bubble", "3D Scatter"]:
                    size = st.selectbox("Размер", [None] + df.select_dtypes(include=np.number).columns.tolist(), key="size")
                else:
                    size = None
                
                if st.button("Создать визуализацию", type="primary", use_container_width=True):
                    st.session_state.viz_requested = True
            
            with cols[1]:
                if 'viz_requested' in st.session_state:
                    with st.spinner("Создаю визуализацию..."):
                        fig = create_visualization(
                            df, viz_type, 
                            x=x_axis, y=y_axis if viz_type != "Тепловая карта" else None,
                            z=z_axis, color=color, size=size
                        )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Дополнительный анализ
                        if viz_type == "Аномалии":
                            anomalies = detect_anomalies(df, x_axis)
                            if len(anomalies) > 0:
                                st.warning(f"Обнаружено {len(anomalies)} аномалий")
                                with st.expander("Показать аномалии"):
                                    st.dataframe(anomalies)
        
        with tab3:
            st.subheader("Автоматический анализ данных")
            
            ai_col1, ai_col2 = st.columns([2, 1])
            
            with ai_col1:
                st.markdown(analyze_with_ai(df))
            
            with ai_col2:
                st.subheader("🤖 AI Инсайты")
                st.markdown(generate_ai_insights(df))
            
            # Анализ временных рядов
            date_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col])]
            if date_cols and len(df.select_dtypes(include=np.number).columns) > 0:
                st.divider()
                st.subheader("⏳ Анализ временных рядов")
                
                ts_col1, ts_col2 = st.columns(2)
                with ts_col1:
                    selected_date = st.selectbox("Временная колонка", date_cols)
                with ts_col2:
                    selected_value = st.selectbox("Анализируемое значение", df.select_dtypes(include=np.number).columns)
                
                if st.button("Проанализировать временной ряд", use_container_width=True):
                    with st.spinner("Выполняю анализ временного ряда..."):
                        ts_fig = time_series_analysis(df, selected_date, selected_value)
                    
                    if ts_fig:
                        st.plotly_chart(ts_fig, use_container_width=True)

else:
    # Страница приветствия
    st.markdown("""
        <div style="text-align:center; padding:50px 20px;">
            <h2>Добро пожаловать в AI Data Analyzer Pro!</h2>
            <p>Загрузите ваш файл с данными, чтобы начать анализ</p>
            <div style="margin:40px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="#4e79a7" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
            </div>
            <div style="display:flex; justify-content:center; gap:20px; margin-top:30px;">
                <div style="border:1px solid #ddd; padding:20px; border-radius:10px; width:200px;">
                    <h3>📊 Визуализация</h3>
                    <p>Более 15 типов интерактивных графиков</p>
                </div>
                <div style="border:1px solid #ddd; padding:20px; border-radius:10px; width:200px;">
                    <h3>🤖 AI Анализ</h3>
                    <p>Автоматическое выявление паттернов и аномалий</p>
                </div>
                <div style="border:1px solid #ddd; padding:20px; border-radius:10px; width:200px;">
                    <h3>⏱️ Быстрый</h3>
                    <p>Оптимизирован для больших данных</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Футер
st.markdown("---")
st.markdown("""
    <div style="text-align:center; padding:20px; color:#666;">
        <p>🏆 AI Data Analyzer Pro | Яковлева Эвелина</p>
        <p>Для работы с большими файлами используйте локальное развертывание</p>
    </div>
""", unsafe_allow_html=True)
