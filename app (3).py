import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import openai
import io
import json
import warnings
import tempfile
import hashlib
import traceback
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.api.types import is_datetime64_any_dtype
import logging

# Настройка логирования
logging.basicConfig(filename='app_errors.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Конфигурация безопасности
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
pd.options.mode.chained_assignment = None  # Отключаем SettingWithCopyWarning

# Глобальные константы
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_ROWS_PREVIEW = 1000
MAX_ROWS_ANALYSIS = 50000
SAMPLE_SIZE = 10000

# Настройки страницы
st.set_page_config(
    page_title="🤖 AI Data Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка API ключа OpenAI из Streamlit Secrets
def get_api_key():
    if 'OPENAI_API_KEY' in st.secrets:
        return st.secrets['OPENAI_API_KEY']
    return st.text_input("Введите OpenAI API ключ:", type="password")

openai.api_key = get_api_key()

# Безопасный CSS
st.markdown("""
    <style>
        .stApp { background-color: #f0f2f6; color: #000000; }
        footer { visibility: hidden; }
        .stAlert { padding: 15px; border-radius: 10px; }
        .st-b7 { color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

# Заголовок с защитой от XSS
st.title("🤖 AI Data Analyzer Pro")
st.markdown("""
    <div style="background-color:#ffffff;padding:10px;border-radius:10px;margin-bottom:20px;">
    <p style="color:#333;font-size:18px;">🚀 <b>Автоматический анализ данных с AI-powered инсайтами</b></p>
    <p style="color:#666;">Загрузите CSV, Excel или JSON — получите полный анализ и визуализацию</p>
    </div>
""", unsafe_allow_html=True)

# Безопасная загрузка данных
@st.cache_data(show_spinner="Загружаю данные... ⏳", ttl=3600, max_entries=3)
def load_data(uploaded_file):
    try:
        # Проверка размера файла
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"Файл слишком большой! Максимальный размер: {MAX_FILE_SIZE//(1024*1024)}MB")
            return None
        
        # Генерация хеша файла для кэширования
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        # Создание безопасного временного файла
        with tempfile.NamedTemporaryFile(delete=True, suffix=uploaded_file.name) as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            tmpfile.flush()
            
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(tmpfile.name, encoding_errors='ignore', on_bad_lines='skip')
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(tmpfile.name)
            elif uploaded_file.name.endswith('.json'):
                with open(tmpfile.name, 'r') as f:
                    data = json.load(f)
                return pd.json_normalize(data)
    except Exception as e:
        logging.error(f"Ошибка загрузки: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Критическая ошибка загрузки: {str(e)}")
        return None

# Оптимизация памяти с защитой
def reduce_mem_usage(df):
    try:
        if df.empty:
            return df
            
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if pd.api.types.is_integer_dtype(col_type):
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
    except Exception as e:
        logging.error(f"Ошибка оптимизации памяти: {str(e)}\n{traceback.format_exc()}")
        return df

# AI анализ данных с защитой от инъекций
@st.cache_data(show_spinner="Анализирую данные... 🔍", ttl=600)
def analyze_with_ai(df):
    try:
        if df.empty or len(df.columns) == 0:
            return "Данные отсутствуют или не содержат колонок"
            
        analysis = f"### 📊 Общий обзор данных\n"
        analysis += f"- **Строки:** {df.shape[0]:,}\n"
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
        logging.error(f"Ошибка анализа: {str(e)}\n{traceback.format_exc()}")
        return f"Ошибка анализа данных: {str(e)}"

# Обнаружение аномалий с защитой
@st.cache_data(show_spinner="Ищу аномалии... 🕵️", ttl=300)
def detect_anomalies(df, column):
    try:
        if df.empty or column not in df.columns:
            return pd.DataFrame()
            
        if len(df) > SAMPLE_SIZE:
            sample = df.sample(min(5000, len(df)))
        else:
            sample = df

        model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        model.fit(sample[[column]])
        df['anomaly'] = model.predict(df[[column]])
        anomalies = df[df['anomaly'] == -1]
        return anomalies
    except Exception as e:
        logging.error(f"Ошибка обнаружения аномалий: {str(e)}\n{traceback.format_exc()}")
        return pd.DataFrame()

# Анализ временных рядов с защитой
@st.cache_data(show_spinner="Анализирую временные ряды... ⏳", ttl=300)
def time_series_analysis(df, date_col, value_col):
    try:
        if df.empty or date_col not in df.columns or value_col not in df.columns:
            return None
            
        df = df.set_index(date_col).sort_index()
        if len(df) > SAMPLE_SIZE:
            df = df.resample('D').mean()

        # Проверка на минимальное количество точек
        if len(df) < 2:
            return None
            
        period = min(12, max(2, len(df)//2))  # Защита от малых периодов
        decomposition = seasonal_decompose(df[value_col], period=period, extrapolate_trend='freq')

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=df.index, y=df[value_col], name='Исходные данные'), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Тренд'), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Сезонность'), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Остатки'), row=4, col=1)

        fig.update_layout(height=800, title_text="Декомпозиция временного ряда")
        return fig
    except Exception as e:
        logging.error(f"Ошибка анализа временных рядов: {str(e)}\n{traceback.format_exc()}")
        return None

# Безопасная генерация инсайтов
@st.cache_data(show_spinner="Генерирую AI инсайты и рекомендации... 🤖", ttl=600)
def generate_ai_insights_and_viz(df):
    if not openai.api_key:
        return "🔑 Ключ OpenAI API не установлен. Добавьте его в Secrets.", None, None, None, None, None, None

    try:
        # Ограничение данных для промпта
        sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
        columns = list(sample_df.columns)
        sample_data = sample_df.head().to_dict()
        
        # Экранирование пользовательских данных
        safe_columns = json.dumps(columns)
        safe_sample = json.dumps(sample_data)

        prompt = (
            f"Ты аналитик данных. Сделай краткий аналитический отчет и дай рекомендации по данным.\n"
            f"Данные: {sample_df.shape[0]} строк, {sample_df.shape[1]} колонок.\n"
            f"Колонки: {safe_columns}.\n"
            f"Первые 5 строк:\n{safe_sample}\n\n"
            f"Напиши инсайты и предложи несколько типов визуализаций из списка:\n"
            f"- гистограмма\n- тепловая карта\n- точечная диаграмма\n- 3D scatter\n"
            f"- временной ряд\n- candlestick\n- аномалии\n- ящик с усами\n"
            f"- столбчатая диаграмма\n- круговая диаграмма\n- scatter с трендом\n"
            f"- pairplot\n- density plot\n- violin plot\n- treemap\n- area chart\n"
            f"- bubble chart\n- 2D histogram\n- boxen plot\n- временной heatmap\n\n"
            f"Ответь ТОЛЬКО в JSON формате:\n"
            f'{{"insights": "...", "visualizations": [{{"viz_type": "...", "x_axis": "...", "y_axis": "...", "z_axis": "...", "color": "...", "size": "..."}}]}}\n'
        )

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты аналитик данных и визуализации. Отвечай ТОЛЬКО в JSON формате."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        text = response['choices'][0]['message']['content'].strip()

        try:
            # Безопасный парсинг JSON
            parsed = json.loads(text)
            insights = parsed.get('insights', '')
            
            # Валидация визуализаций
            visualizations = parsed.get('visualizations', [])
            if visualizations:
                viz = visualizations[0]
                return (
                    insights,
                    viz.get('viz_type', None),
                    viz.get('x_axis', None),
                    viz.get('y_axis', None),
                    viz.get('z_axis', None),
                    viz.get('color', None),
                    viz.get('size', None)
                )
            return insights, None, None, None, None, None, None
        except json.JSONDecodeError:
            # Возвращаем сырой текст если JSON невалиден
            return text, None, None, None, None, None, None

    except Exception as e:
        logging.error(f"Ошибка OpenAI: {str(e)}\n{traceback.format_exc()}")
        return f"Ошибка вызова OpenAI API: {str(e)}", None, None, None, None, None, None

# Безопасная визуализация данных
def create_visualization(df, viz_type, x=None, y=None, z=None, color=None, size=None):
    try:
        if df.empty or viz_type is None:
            return None
            
        # Защита от больших датасетов
        viz_df = df.sample(min(SAMPLE_SIZE, len(df))) if len(df) > SAMPLE_SIZE else df
        
        fig = None

        if viz_type == "Гистограмма" and x in viz_df.columns:
            fig = px.histogram(viz_df, x=x, color=color, marginal="box", nbins=50)

        elif viz_type == "Тепловая карта":
            num_cols = viz_df.select_dtypes(include=np.number).columns
            if len(num_cols) > 1:
                corr = viz_df[num_cols].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Корреляционная матрица")

        elif viz_type == "3D scatter" and all(col in viz_df.columns for col in [x, y, z]):
            fig = px.scatter_3d(viz_df, x=x, y=y, z=z, color=color, size=size)

        elif viz_type == "Временной ряд" and x in viz_df.columns and y in viz_df.columns:
            if is_datetime64_any_dtype(viz_df[x]):
                fig = px.line(viz_df.sort_values(x), x=x, y=y, color=color)

        elif viz_type == "candlestick":
            required = {'open', 'high', 'low', 'close', x}
            if set(required).issubset(set(viz_df.columns)):
                fig = go.Figure(data=[go.Candlestick(
                    x=viz_df[x],
                    open=viz_df['open'],
                    high=viz_df['high'],
                    low=viz_df['low'],
                    close=viz_df['close']
                )])

        elif viz_type == "аномалии" and x in viz_df.columns and y in viz_df.columns:
            anomalies = detect_anomalies(viz_df, y)
            if not anomalies.empty:
                fig = px.scatter(viz_df, x=x, y=y, title="Аномалии в данных")
                fig.add_trace(go.Scatter(
                    x=anomalies[x], 
                    y=anomalies[y], 
                    mode='markers',
                    marker=dict(color='red', size=10), 
                    name='Аномалии'
                ))

        elif viz_type == "точечная диаграмма" and x in viz_df.columns and y in viz_df.columns:
            fig = px.scatter(viz_df, x=x, y=y, color=color, size=size)

        if fig:
            fig.update_layout(height=600, margin=dict(t=50, b=50, l=50, r=50))
            return fig
        return None

    except Exception as e:
        logging.error(f"Ошибка визуализации: {str(e)}\n{traceback.format_exc()}")
        return None

# --- Основной интерфейс с защитой ---

uploaded_file = st.file_uploader(
    "Загрузите файл с данными (.csv, .xlsx, .json)", 
    type=["csv", "xlsx", "xls", "json"],
    accept_multiple_files=False
)

if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"Файл слишком большой! Максимальный размер: {MAX_FILE_SIZE//(1024*1024)}MB")
    else:
        df = load_data(uploaded_file)
        if df is not None:
            if not df.empty:
                df = reduce_mem_usage(df)
                
                st.subheader("Предварительный просмотр данных")
                st.dataframe(df.head(min(MAX_ROWS_PREVIEW, len(df))))
                
                # Защита от больших датасетов
                if len(df) > MAX_ROWS_ANALYSIS:
                    st.warning(f"⚠️ Датсет слишком большой для полного анализа. Будут использованы первые {MAX_ROWS_ANALYSIS} строк.")
                    df = df.head(MAX_ROWS_ANALYSIS)
                
                # AI анализ
                insights, viz_type, x_axis, y_axis, z_axis, color, size = generate_ai_insights_and_viz(df)
                
                st.subheader("🤖 AI инсайты")
                st.markdown(insights if insights else "Нет данных для инсайтов.")
                
                # Визуализация
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
                    
                    # Анализ аномалий
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if num_cols:
                        col_for_anom = st.selectbox("Выберите числовой столбец для поиска аномалий", num_cols)
                        anomalies = detect_anomalies(df, col_for_anom)
                        if not anomalies.empty:
                            st.write(f"Найдено {len(anomalies)} аномалий в столбце {col_for_anom}")
                            st.dataframe(anomalies.head(20))
                            fig_anom = px.scatter(df, x=df.index, y=col_for_anom, title=f"Аномалии в {col_for_anom}")
                            fig_anom.add_trace(go.Scatter(
                                x=anomalies.index, 
                                y=anomalies[col_for_anom],
                                mode='markers', 
                                marker=dict(color='red', size=10),
                                name='Аномалии'
                            ))
                            st.plotly_chart(fig_anom, use_container_width=True)
                    
                    # Анализ временных рядов
                    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
                    if not date_cols:
                        # Безопасная конвертация
                        for c in df.columns:
                            try:
                                df[c] = pd.to_datetime(df[c], errors='coerce')
                                if not df[c].isnull().all():
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
                st.error("Загруженный файл не содержит данных или произошла ошибка обработки")
else:
    st.info("Пожалуйста, загрузите файл с данными для анализа.")

# Защита от XSS в выводе
st.markdown("""
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const userContentElements = document.querySelectorAll('.stMarkdown');
            userContentElements.forEach(el => {
                el.innerHTML = el.textContent;
            });
        });
    </script>
""", unsafe_allow_html=True)
