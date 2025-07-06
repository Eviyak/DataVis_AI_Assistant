import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fpdf import FPDF
import io

st.set_page_config(page_title="📊 AI Визуализатор Данных", layout="wide")
st.title("🤖 AI-помощник для визуализации и анализа данных")
st.markdown("Загрузите CSV/Excel файл, чтобы получить визуализацию, автоанализ и PDF-отчёт")

# 📄 Генерация PDF
def generate_pdf_report(df, summary_text):
    pdf = FPDF()
    pdf.add_page()

    # ✅ Добавляем шрифт Unicode
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.cell(200, 10, txt="📄 Автоматический отчёт", ln=True, align="C")
    pdf.ln(10)

    for line in summary_text.split("\n"):
        pdf.multi_cell(0, 10, txt=line)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# 📂 Загрузка файла
uploaded_file = st.file_uploader("Загрузите CSV или Excel файл", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"Успешно загружено {df.shape[0]} строк и {df.shape[1]} колонок")

        st.subheader("🔍 Просмотр данных")
        st.dataframe(df.head())

        st.subheader("📈 Визуализация числовых признаков")
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols):
            col = st.selectbox("Выберите числовой столбец", num_cols)
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[col], ax=ax[0], kde=True)
            sns.boxplot(x=df[col], ax=ax[1])
            st.pyplot(fig)

        st.subheader("🤖 ML-анализ (RandomForest)")
        target = st.selectbox("Выберите целевую колонку (таргет)", df.columns)
        X = df.drop(columns=[target])
        y = df[target]
        X = pd.get_dummies(X)  # авто-обработка категорий

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        report = classification_report(y_test, preds)

        st.code(report, language='text')

        st.subheader("📄 Скачать отчёт в PDF")
        summary = f"""Отчёт по датасету:
- Строк: {df.shape[0]}
- Колонок: {df.shape[1]}
- Целевая переменная: {target}

Метрика модели:
{report}
        """
        pdf_file = generate_pdf_report(df, summary)
        st.download_button("📥 Скачать PDF отчёт", pdf_file, file_name="report.pdf")

    except Exception as e:
        st.error(f"Ошибка: {e}")

else:
    st.info("Пожалуйста, загрузите файл")
