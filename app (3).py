import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import io
from io import BytesIO
import tempfile
from fpdf import FPDF

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

def generate_pdf_report(data_info, stats_info, ai_info, hist_img_buf=None, boxplot_img_buf=None):
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

    # Сохраняем графики во временные файлы для вставки
    if hist_img_buf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_hist:
            tmp_hist.write(hist_img_buf.getbuffer())
            tmp_hist.flush()
            pdf.image(tmp_hist.name, x=10, w=90)

    if boxplot_img_buf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_box:
            tmp_box.write(boxplot_img_buf.getbuffer())
            tmp_box.flush()
            pdf.image(tmp_box.name, x=110, w=90)

    pdf.ln(5)

    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(0, 10, 'AI-модель', ln=True)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 8, ai_info if ai_info else "Информация отсутствует")

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return io.BytesIO(pdf_bytes)

# ==== Основной интерфейс ====

uploaded_file = st.file_uploader("Загрузите файл", type=["csv", "xlsx", "xls", "json"])

ai_report_text = None
hist_buf = None
boxplot_buf = None
selected = None

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

                fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                sns.histplot(df[selected], ax=ax_hist, kde=True)
                ax_hist.set_title("Распределение")
                hist_buf = fig_to_bytes(fig_hist)

                fig_box, ax_box = plt.subplots(figsize=(6, 4))
                sns.boxplot(x=df[selected], ax=ax_box)
                ax_box.set_title("Boxplot")
                boxplot_buf = fig_to_bytes(fig_box)
            else:
                st.warning("Нет числовых колонок для визуализации.")

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
                    st.error(f"Ошибка обучения модели: {e}")
            else:
                st.warning("Недостаточно числовых признаков для обучения модели.")

        with tab4:
            st.subheader("📄 Генерация PDF-отчёта")
            data_info = f"Количество строк: {df.shape[0]}\nКоличество колонок: {df.shape[1]}"
            stats_summary = f"""
Выбранная колонка для визуализации: {selected if selected else 'не выбрана'}

Основные статистики по числовым данным:
{df.describe().to_string()}
"""

            if st.button("📥 Скачать отчёт в PDF"):
                pdf_buffer = generate_pdf_report(data_info, stats_summary, ai_report_text, hist_buf, boxplot_buf)
                st.download_button("📄 Скачать PDF", data=pdf_buffer, file_name="ai_data_report.pdf", mime="application/pdf")
else:
    st.info("Пожалуйста, загрузите CSV, Excel или JSON файл для анализа.")
