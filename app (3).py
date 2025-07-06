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

def generate_pdf_report(data_info, stats_info, ai_info, hist_img_buf=None, boxplot_img_buf=None):
    pdf = FPDF()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)

    pdf.add_page()

    pdf.set_font('DejaVu', 'B', 20)
    pdf.cell(0, 15, 'Отчёт', align='C', ln=True)
    pdf.ln(10)

    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(0, 10, 'Данные
