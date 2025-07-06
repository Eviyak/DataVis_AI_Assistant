import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import io
from io import BytesIO
from fpdf import FPDF
import tempfile

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="📊 AI Визуализатор Данных", layout="wide")
st.title("📊 AI-помощник для визуализации и анализа данных")
st.markdown("Загрузите файл (CSV, Excel или JSON) — и получите автоматический анализ + графики + AI классификацию + PDF отчёт.")

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
        st.error(f"Ошибка при загрузке файла: {e}")
        return None

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_pdf_report(data_info, stats_info, ai_info, images):
    pdf = FPDF()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
    pdf.add_page()

    pdf.set_font('DejaVu', 'B', 20)
    pdf.cell(0, 15, 'Отчёт', align='C', ln=True)
    pdf.ln(10)

    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(0, 10, 'Данные', ln=True)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 8, data_info)
    pdf.ln(5)

    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(0, 10, 'Статистика', ln=True)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 8, stats_info)
    pdf.ln(5)

    for img_buf in images:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(img_buf.getbuffer())
            tmp.flush()
            pdf.image(tmp.name, x=10, w=190)

    pdf.ln(5)

    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(0, 10, 'AI-модель', ln=True)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 8, ai_info if ai_info else "Информация отсутствует")

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

uploaded_file = st.file_uploader("Загрузите файл", type=["csv", "xlsx", "xls", "json"])

ai_report_text = None
images = []

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.success(f"✅ Загружено {df.shape[0]} строк и {df.shape[1]} колонок")

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Данные", "📈 Графики", "🧠 AI-модель", "📄 Отчёт"])

        with tab1:
            st.dataframe(df.head(100))

        with tab2:
            st.subheader("📈 Автоматическая визуализация")
            num_cols = df.select_dtypes(include='number').columns
            cat_cols = df.select_dtypes(include='object').columns

            if len(num_cols) > 0:
                st.markdown("### Гистограммы")
                for col in num_cols:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col], kde=True, ax=ax)
                    ax.set_title(f"Гистограмма: {col}")
                    st.pyplot(fig)
                    images.append(fig_to_bytes(fig))

            if len(cat_cols) > 0:
                st.markdown("### Распределение категорий")
                for col in cat_cols:
                    fig, ax = plt.subplots()
                    df[col].value_counts().plot(kind='bar', ax=ax)
                    ax.set_title(f"Категориальное распределение: {col}")
                    st.pyplot(fig)
                    images.append(fig_to_bytes(fig))

            if len(num_cols) > 1:
                st.markdown("### Корреляционная матрица")
                fig, ax = plt.subplots()
                sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                ax.set_title("Корреляционная матрица")
                st.pyplot(fig)
                images.append(fig_to_bytes(fig))

                st.markdown("### Парные диаграммы")
                pairplot = sns.pairplot(df[num_cols])
                st.pyplot(pairplot)

        with tab3:
            st.subheader("🧠 Обучение модели (RandomForestClassifier)")
            target_column = st.selectbox("Выберите целевую переменную (классификация)", df.columns)
            features = [col for col in df.select_dtypes(include='number').columns if col != target_column]

            if len(features) > 0:
                X = df[features]
                y = df[target_column]

                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    ai_report_text = classification_report(y_test, y_pred, zero_division=0)
                    st.code(ai_report_text, language='text')
                except Exception as e:
                    st.error(f"Ошибка обучения: {e}")
            else:
                st.warning("Недостаточно числовых признаков для обучения модели.")

        with tab4:
            st.subheader("📄 Генерация PDF-отчёта")
            data_info = f"Количество строк: {df.shape[0]}\nКоличество колонок: {df.shape[1]}"
            stats_summary = f"""
Основные статистики по числовым данным:
{df.describe().to_string()}
"""
            if st.button("📥 Скачать отчёт в PDF"):
                pdf_buffer = generate_pdf_report(data_info, stats_summary, ai_report_text, images)
                st.download_button("📄 Скачать PDF", data=pdf_buffer, file_name="ai_data_report.pdf", mime="application/pdf")
else:
    st.info("Пожалуйста, загрузите CSV, Excel или JSON файл для анализа.")
