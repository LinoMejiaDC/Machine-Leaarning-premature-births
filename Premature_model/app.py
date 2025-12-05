import streamlit as st
import pandas as pd
import pickle
import os
import sys

# --- Ajustar path para importar scripts (se precisar) ---
# Adiciona o diretório atual ao PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

# try:
#     from scripts.Wrangling import (
#         # importe aqui funções que você quiser usar para pré-processar
#         # ex: fill_missing_with_mode, categorize_columns, etc.
#         )
    
# except Exception:
#     # Se não precisar dos scripts agora, pode ignorar
#     pass

# --- Carregar modelo ---
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "propensity_model.pkl")

#@st.cache_resource
# def load_model():
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     return model

# model = load_model()

# --- UI ---
st.title("Premature Birth Risk  Scoring Tool")
st.write("Preencha os dados da paciente para calcular o risco.")

# Exemplos de variáveis de entrada – adapte para o que seu modelo usa
age = st.number_input("Idade (anos)", min_value=10, max_value=60, value=28)
education = st.selectbox(
    "Escolaridade",
    ["Primary", "Secondary", "High School", "University", "Postgraduate"]
)
bmi = st.number_input("BMI", min_value=10.0, max_value=45.0, value=23.5)
parity = st.number_input("Número de gestações prévias (parity)", min_value=0, max_value=10, value=1)

# Mapear escolaridade para código numérico, se seu modelo usa assim
education_map = {
    "Primary": 1,
    "Secondary": 2,
    "High School": 3,
    "University": 4,
    "Postgraduate": 5
}
education_code = education_map[education]

# Quando clicar no botão, monta o dataframe e faz o score
if st.button("Calcular risco de parto prematuro"):
    # Monte o DF com os MESMOS nomes de colunas usados no treino
    input_data = pd.DataFrame([{
        "AGE": age,
        "EDUCATION": education_code,
        "BMI": bmi,
        "PARITY": parity
    }])

    # Se você tiver um pipeline completo salvo (pré-processamento + modelo),
    # aqui já funciona direto. Caso contrário, chame funções de Wrangling.

    # prob = model.predict_proba(input_data)[:, 1][0]
    prob= 0.33

    st.markdown(f"### Risco estimado: **{prob:.2%}**")

    # Opcional: classificação por faixas
    if prob >= 0.7:
        st.error("Alto risco – atenção especial recomendada.")
    elif prob >= 0.4:
        st.warning("Risco moderado.")
    else:
        st.success("Baixo risco.")
