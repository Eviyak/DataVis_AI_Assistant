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
warnings.filterwarnings('ignore')

# Настройки страницы
st.set_page_config(
    page_title="🤖 AI Data Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Темная/светлая тема
theme = st.sidebar.radio("🎨 Тема", ["Светлая", "Темная"])
if theme == "Темная":
    plt.style.use('dark_background')
    st.markdown("""
        <style>
            .stApp { background-color: #1E1E1E; }
            .css-1d391kg { color: white; }
        </style>
    """, unsafe_allow_html=True)

# Заголовок
st.title("🤖 AI Data Analyzer Pro")
st.markdown("""
    *Автоматический анализ данных с AI-powered инсайтами*  
    **Загрузите CSV, Excel или JSON** — получите полный анализ и визуализацию
""")

# Функция загрузки данных
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
        
        # Автоопределение дат
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки: {str(e)}")
        return None

# AI анализ данных
def analyze_with_ai(df):
    try:
        analysis = f"Датасет содержит {df.shape[0]} строк и {df.shape[1]} колонок.\n"
        
        # Числовые колонки
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            analysis += "\n**Числовые данные:**\n"
            for col in num_cols:
                analysis += f"- {col}: среднее = {df[col].mean():.2f}, мин = {df[col].min():.2f}, макс = {df[col].max():.2f}\n"
        
        # Категориальные колонки
        cat_cols = df.select_dtypes(exclude=np.number).columns
        if len(cat_cols) > 0:
            analysis += "\n**Категориальные данные:**\n"
            for col in cat_cols:
                analysis += f"- {col}: {df[col].nunique()} уникальных значений\n"
        
        # Пропущенные значения
        missing = df.isnull().sum()
        if missing.sum() > 0:
            analysis += "\n⚠️ **Пропущенные значения:**\n"
            for col, count in missing.items():
                if count > 0:
                    analysis += f"- {col}: {count} пропусков ({count/len(df):.1%})\n"
        
        # Корреляции
        if len(num_cols) > 1:
            corr = df[num_cols].corr().abs().unstack().sort_values(ascending=False)
            strong_corr = corr[(corr > 0.7) & (corr < 1)]
            if len(strong_corr) > 0:
                analysis += "\n🔗 **Сильные корреляции:**\n"
                for pair, value in strong_corr.items():
                    analysis += f"- {pair[0]} и {pair[1]}: {value:.2f}\n"
        
        return analysis
    except Exception as e:
        return f"Ошибка анализа: {str(e)}"

# Обнаружение аномалий
def detect_anomalies(df, column):
    try:
        model = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = model.fit_predict(df[[column]])
        anomalies = df[df['anomaly'] == -1]
        return anomalies
    except:
        return None

# Временной анализ
def time_series_analysis(df, date_col, value_col):
    try:
        df = df.set_index(date_col).sort_index()
        decomposition = seasonal_decompose(df[value_col], period=12)
        
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
def generate_ai_insights(df):
    try:
        # Для демо используем локальную логику вместо реального API вызова
        insights = ["AI анализ выявил следующие ключевые моменты:"]
        
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 1:
            corr = df[num_cols].corr().abs().unstack().sort_values(ascending=False)
            top_corr = corr[(corr > 0.5) & (corr < 1)].head(1)
            if len(top_corr) > 0:
                pair, value = top_corr.index[0], top_corr.values[0]
                insights.append(f"Обнаружена сильная корреляция ({value:.2f}) между '{pair[0]}' и '{pair[1]}'")
        
        date_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col])]
        if date_cols and len(num_cols) > 0:
            insights.append(f"Данные содержат временные метки в колонке '{date_cols[0]}' - рекомендуется анализ временных рядов")
        
        missing = df.isnull().sum()
        if missing.sum() > 0:
            worst_col = missing.idxmax()
            insights.append(f"⚠️ Колонка '{worst_col}' содержит {missing.max()} пропущенных значений ({missing.max()/len(df):.1%} данных)")
        
        return "\n\n".join(insights)
    except:
        return "Не удалось сгенерировать AI инсайты"

# Визуализации
def create_visualization(df, viz_type, x=None, y=None, z=None, color=None, size=None, animation=None):
    try:
        if viz_type == "Гистограмма":
            fig = px.histogram(df, x=x, color=color, marginal="box", nbins=50)
        
        elif viz_type == "Тепловая карта":
            corr = df.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')
        
        elif viz_type == "3D Scatter":
            fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, size=size)
        
        elif viz_type == "Аномалии":
            anomalies = detect_anomalies(df, x)
            fig = px.scatter(df, x=x, y=y, color=df.index.isin(anomalies.index))
        
        elif viz_type == "Временной ряд":
            df = df.set_index(x).sort_index()
            fig = px.line(df, y=y, color=color)
            if len(df) > 30:
                df['rolling'] = df[y].rolling(7).mean()
                fig.add_scatter(x=df.index, y=df['rolling'], name='Скользящее среднее (7)')
        
        elif viz_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=df[x],
                open=df[y],
                high=df[y]+df[y].std(),
                low=df[y]-df[y].std(),
                close=df[y])])
        
        else:
            fig = px.scatter(df, x=x, y=y, color=color, size=size, animation_frame=animation)
        
        fig.update_layout(
            template="plotly_dark" if theme == "Темная" else "plotly_white",
            hovermode="x unified"
        )
        return fig
    except Exception as e:
        st.error(f"Ошибка визуализации: {str(e)}")
        return None

# Основной интерфейс
uploaded_file = st.sidebar.file_uploader("📤 Загрузите файл", type=["csv", "xlsx", "json"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.sidebar.success(f"✅ Успешно загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
        
        # Основные вкладки
        tab1, tab2, tab3 = st.tabs(["🔍 Обзор данных", "📊 Визуализация", "🤖 AI Анализ"])
        
        with tab1:
            st.subheader("Превью данных")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("Техническая информация")
            st.write(f"**Типы данных:**\n{df.dtypes.to_frame().T}")
            
            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.warning(f"⚠️ Пропущенные значения:\n{missing[missing > 0]}")
            else:
                st.success("✅ Нет пропущенных значений")
        
        with tab2:
            viz_type = st.selectbox("Тип визуализации", [
                "Гистограмма", "Тепловая карта", "3D Scatter", 
                "Аномалии", "Временной ряд", "Candlestick"
            ])
            
            cols = st.columns(2)
            with cols[0]:
                x_axis = st.selectbox("Ось X", df.columns, index=0)
            with cols[1]:
                y_axis = st.selectbox("Ось Y", df.columns, index=min(1, len(df.columns)-1))
            
            extra_cols = st.columns(3)
            with extra_cols[0]:
                z_axis = st.selectbox("Ось Z (3D)", [None] + df.select_dtypes(include=np.number).columns.tolist())
            with extra_cols[1]:
                color = st.selectbox("Цвет", [None] + df.columns.tolist())
            with extra_cols[2]:
                size = st.selectbox("Размер", [None] + df.select_dtypes(include=np.number).columns.tolist())
            
            if st.button("Создать визуализацию"):
                fig = create_visualization(
                    df, viz_type, 
                    x=x_axis, y=y_axis, z=z_axis,
                    color=color, size=size
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Дополнительный анализ
                    if viz_type == "Аномалии":
                        anomalies = detect_anomalies(df, x_axis)
                        if len(anomalies) > 0:
                            st.warning(f"Обнаружено {len(anomalies)} аномалий")
                            st.dataframe(anomalies)
        
        with tab3:
            st.subheader("Автоматический анализ")
            st.write(analyze_with_ai(df))
            
            st.subheader("AI Инсайты")
            st.write(generate_ai_insights(df))
            
            # Анализ временных рядов
            date_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col])]
            if date_cols and len(df.select_dtypes(include=np.number).columns) > 0:
                st.subheader("Анализ временных рядов")
                selected_date = st.selectbox("Выберите временную колонку", date_cols)
                selected_value = st.selectbox("Выберите значение", df.select_dtypes(include=np.number).columns)
                
                if st.button("Проанализировать временной ряд"):
                    ts_fig = time_series_analysis(df, selected_date, selected_value)
                    if ts_fig:
                        st.plotly_chart(ts_fig, use_container_width=True)
            
            # Прогнозирование
            if len(date_cols) > 0 and len(df.select_dtypes(include=np.number).columns) > 0:
                st.subheader("Прогнозирование")
                if st.button("Сделать прогноз (линейная регрессия)"):
                    try:
                        model = LinearRegression()
                        X = pd.to_numeric(df[date_cols[0]]).values.reshape(-1, 1)
                        y = df[selected_value]
                        model.fit(X, y)
                        
                        future_dates = pd.date_range(
                            start=df[date_cols[0]].max(),
                            periods=10,
                            freq=pd.infer_freq(df[date_cols[0]])
                        
                        future_X = pd.to_numeric(future_dates).values.reshape(-1, 1)
                        future_y = model.predict(future_X)
                        
                        fig = go.Figure()
                        fig.add_scatter(x=df[date_cols[0]], y=y, name="Исторические данные")
                        fig.add_scatter(x=future_dates, y=future_y, name="Прогноз", line=dict(dash='dot'))
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.error("Ошибка при прогнозировании")

else:
    st.info("👈 Пожалуйста, загрузите файл для начала анализа")
    
st.markdown("---")
st.markdown("""
    ### *Яковлева Эвелина Вячеслаовна*
""")
