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

def generate_pdf_report(df, stats_text, ai_report, hist_fig_buf=None, boxplot_fig_buf=None):
    pdf = FPDF()
    pdf.add_font('DejaVu', '', os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf'), uni=True)
    pdf.set_font('DejaVu', '', 16)

    # Первая страница с заголовком "Отчет"
    pdf.add_page()
    pdf.set_xy(0, 50)
    pdf.cell(210, 10, "Отчет", ln=True, align='C')

    # Вторая страница - Данные
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, "Данные", ln=True)
    pdf.set_font('DejaVu', '', 12)
    data_info = f"Файл содержит {df.shape[0]} строк и {df.shape[1]} колонок.\n\nТипы данных:\n{df.dtypes.to_string()}\n\nПропущенные значения:\n{df.isnull().sum().to_string()}"
    pdf.multi_cell(0, 8, data_info)

    # Третья страница - Статистика + графики
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, "Статистика", ln=True)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 8, stats_text)

    # Вставляем графики если они есть
    y_pos = pdf.get_y() + 10
    x_center = 10
    img_w = 90  # ширина изображения

    if hist_fig_buf:
        pdf.image(hist_fig_buf, x=x_center, y=y_pos, w=img_w)
    if boxplot_fig_buf:
        # Вторая картинка справа
        pdf.image(boxplot_fig_buf, x=x_center + img_w + 10, y=y_pos, w=img_w)

    # Четвёртая страница - AI модель
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, "AI-модель", ln=True)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 8, ai_report if ai_report else "Модель не обучалась.")

    # Генерируем PDF в байтах и возвращаем буфер
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    buffer = io.BytesIO(pdf_bytes)
    buffer.seek(0)
    return buffer

uploaded_file = st.file_uploader("Загрузите файл", type=["csv", "xlsx", "xls", "json"])

ai_report_text = None
hist_buf = None
boxplot_buf = None

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

                # Сохраняем графики для отчёта
                hist_buf = fig_to_bytes(fig=fig)
                # Разделяем изображение для вставки в PDF, потому что обе картинки на одном рисунке
                # Для простоты сделаем отдельные графики отдельно:
                # Гистограмма
                fig_hist, ax_hist = plt.subplots(figsize=(6,4))
                sns.histplot(df[selected], ax=ax_hist, kde=True)
                ax_hist.set_title("Распределение")
                hist_buf = fig_to_bytes(fig_hist)
                # Боксплот
                fig_box, ax_box = plt.subplots(figsize=(6,4))
                sns.boxplot(x=df[selected], ax=ax_box)
                ax_box.set_title("Boxplot")
                boxplot_buf = fig_to_bytes(fig_box)

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
            stats_summary = f"""
Выбранная колонка для визуализации: {selected if 'selected' in locals() else 'не выбрана'}

Основные статистики по числовым данным:
{df.describe().to_string()}
"""
            if st.button("📥 Скачать отчёт в PDF"):
                pdf_buffer = generate_pdf_report(df, stats_summary, ai_report_text, hist_buf, boxplot_buf)
                st.download_button("📄 Скачать PDF", data=pdf_buffer, file_name="ai_data_report.pdf", mime="application/pdf")
else:
    st.info("Пожалуйста, загрузите CSV, Excel или JSON файл для анализа.")
