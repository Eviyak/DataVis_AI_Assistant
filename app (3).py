import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import io
from io import BytesIO
from fpdf import FPDF
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Настройка страницы
st.set_page_config(page_title="📊 AI Визуализатор Данных", layout="wide")
st.title("📊 AI-помощник для визуализации и анализа данных")
st.markdown("Загрузите файл (CSV, Excel или JSON) — и получите автоматический анализ + графики + AI классификацию + PDF отчёт.")

# Загрузка и парсинг файла
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

# 📄 Генерация PDF
def generate_pdf_report(df, summary_text):
    pdf = FPDF()
    # Загружаем шрифт из файла DejaVuSans.ttf, который должен лежать в той же папке, что и скрипт
    pdf.add_font('DejaVu', '', os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf'), uni=True)
    pdf.set_font('DejaVu', '', 14)

    pdf.add_page()
    pdf.cell(0, 10, 'Отчет по данным', ln=True)

    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 10, summary_text)

    # Генерируем PDF как строку, затем конвертим в байты и помещаем в BytesIO
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    buffer = io.BytesIO(pdf_bytes)
    buffer.seek(0)
    return buffer
    
# Интерфейс загрузки
uploaded_file = st.file_uploader("Загрузите файл", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.success(f"✅ Загружено {df.shape[0]} строк и {df.shape[1]} колонок")

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Данные", "📈 Анализ", "🧠 AI-модель", "📄 Отчёт"])

        with tab1:
            st.dataframe(df.head(100))

        with tab2:
            st.subheader("📊 Статистика")
            st.write("Типы данных:")
            st.write(df.dtypes)
            st.write("Пропущенные значения:")
            st.write(df.isnull().sum())

            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols) > 0:
                selected = st.selectbox("Выберите числовую колонку для гистограммы", num_cols)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                sns.histplot(df[selected], ax=ax1, kde=True)
                ax1.set_title("Распределение")
                sns.boxplot(x=df[selected], ax=ax2)
                ax2.set_title("Boxplot")
                st.pyplot(fig)

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
                    report = classification_report(y_test, y_pred, zero_division=0)
                    st.code(report, language='text')
                except Exception as e:
                    st.error(f"Ошибка обучения: {e}")
            else:
                st.warning("Недостаточно числовых признаков для обучения модели.")

        with tab4:
            st.subheader("📄 Генерация PDF-отчёта")
            report_summary = f"""
Файл: {uploaded_file.name}
Строк: {df.shape[0]}, Колонок: {df.shape[1]}

Типы данных:
{df.dtypes.to_string()}

Пропущенные значения:
{df.isnull().sum().to_string()}

(Если обучалась AI-модель, см. вкладку 'AI-модель')
"""
            if st.button("📥 Скачать отчёт в PDF"):
                pdf = generate_pdf_report(df, report_summary)
                st.download_button("📄 Скачать PDF", data=pdf, file_name="ai_data_report.pdf", mime="application/pdf")
else:
    st.info("Пожалуйста, загрузите CSV, Excel или JSON файл для анализа.")
