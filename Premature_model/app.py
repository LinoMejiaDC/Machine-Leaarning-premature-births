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

# 1-numerico 
#age = st.number_input("Idade (anos)", min_value=10, max_value=60, value=28)
REC94_S410B = st.number_input("¿Cuántos meses de embarazo tenía la última vez que recibió control prenatal?", 
                              min_value=1, max_value=50, value=1)


# 2-Box
REC41_M44_label = st.selectbox("¿Le dijeron a dónde acudir si llegaba a presentar estas complicaciones?",
                         ["Si", "No", "No sabe"])
REC41_M44_map = {"Si": 0.19000867069771954,"No": 0.19639551643283246,"No sabe": 0.0}
REC41_M44 = REC41_M44_map[REC41_M44_label]


# REC41_M14
REC41_M14 = st.number_input(" ¿Cuántos controles prenatales tuvo Ud. durante el embarazo?", 
                              min_value=1, max_value=50, value=1)


#REC94_S411J
REC94_S411J_label = st.selectbox("¿En alguno de sus controles: Le informaron sobre sus derechos?",
                         ["Si", "No", "No sabe"])
REC94_S411J_map = {"Si":  0.18828486477211587,"No": 0.21529530448512416,"No sabe": 0.2683466839607115}
REC94_S411J = REC94_S411J_map[REC94_S411J_label]


# REC94_QI422A_B
REC94_QI422A_B_label = st.selectbox("¿Durante el embarazo le diagnosticaron o le dijeron que tenía anemia?",
                         ["Si", "No", "No sabe"])
REC94_QI422A_B_map = {"Si":  0.20122936856588078,"No": 0.18718742153602874,"No sabe": 0.15168244979533013}
REC94_QI422A_B = REC94_QI422A_B_map[REC94_QI422A_B_label]


# RE516171_V631
RE516171_V631_label = st.selectbox("Si en las semanas siguientes, Ud. descubriera que está embarazada, ¿para Ud. sería un gran problema, un pequeño problema o no sería problema?",
                         ["Un gran problema", "Un pequeño problema", "No seria problema", "No puede quedar embarazada/No tiene relaciones sexuales"])
RE516171_V631_map = {"Un gran problema":  0.09738093049305999,
                     "Un pequeño problema": 0.10542367650484936,
                     "No seria problema": 0.11905953037753354,
                     "No puede quedar embarazada/No tiene relaciones sexuales": 0.0409690398458394}
RE516171_V631 = RE516171_V631_map[RE516171_V631_label]


#REC94_S411K
REC94_S411K_label = st.selectbox("¿Le enseñaron como preparar pezones para lactancia materna?",
                         ["Si", "No", "No sabe"])
REC94_S411K_map = {"Si":  0.17941727752317377,"No": 0.2269354834251489,"No sabe": 0.26442103024491526}
REC94_S411K = REC94_S411K_map[REC94_S411K_label]

#RE516171_V504
RE516171_V504_label = st.selectbox("¿Su esposo /compañero vive con usted ahora o permanece en otro sitio?",
                         ["Vive con el/ella", "Vive en otro sitio"])
RE516171_V504_map = {"Vive con el/ella":  0.10208848519056772,"Vive en otro sitio": 0.1240989476333886}
RE516171_V504 = RE516171_V504_map[RE516171_V504_label]

#REC41_M42C
REC41_M42C_label = st.selectbox("¿Durante su embarazo en alguno de sus controles: Le tomaron le presión arterial?",
                         ["Si", "No", "No sabe"])
REC41_M42C_map = {"Si":  0.19228437721009464,"No": 0.21348426846751073,"No sabe": 0.28799041593963065}
REC41_M42C = REC41_M42C_map[REC41_M42C_label]

# REC94_S411I
REC94_S411I_label = st.selectbox("¿En alguno de sus controles: Le informaron como alimentarse?",
                         ["Si", "No", "No sabe"])
REC94_S411I_map = {"Si":  0.19061925796941065,"No": 0.23943979749526043,"No sabe": 0.19876263071487615}
REC94_S411I = REC94_S411I_map[REC94_S411I_label]



# RE516171_V743A
RE516171_V743A_label = st.selectbox("Quién tiene la última palabra sobre: cuidado de su salud",
                         ["Nadie", 
                        "Entrevistada",
                        "Entrevistada y esposo/compañero",
                        "Entrevistada y otra persona",
                        "Solo esposo/compañero",
                        "Alguien más"])
RE516171_V743A_map = {"Nadie":  0.2419802462791059,
                     "Entrevistada": 0.09833040685696708,
                     "Entrevistada y esposo/compañero": 0.12964322973686435, 
                     "Entrevistada y otra persona":0.0570099130018505 ,
                     "Solo esposo/compañero": 0.11117081645419798,
                     "Alguien más":0.1923549879480032}
RE516171_V743A = RE516171_V743A_map[RE516171_V743A_label]

# REC91_V208
REC91_V208 = st.number_input(" ¿Cuántos nacimientos en los últimos cinco años ?", 
                              min_value=1, max_value=50, value=1)

REC91_V212 = st.number_input("¿Cual edad del entrevistado al primer nacimiento?", 
                              min_value=1, max_value=100, value=1)



CSALUD01_QS25BB_label = st.selectbox("Por sus antepasados y de acuerdo a sus costumbres, usted se siente o considera:",
                         ["Quechua", 
                        "Aimara",
                        "Nativo o indígena de la amazonia",
                        "Perteneciente o parte de otro pueblo",
                        "Negro/moreno/zambo/mulato/pueblo afroperuano o afrodescendiente",
                        "Blanco", 
                        "Mestizo",
                        "Otro",
                        "No sabe"                        
                        ])
CSALUD01_QS25BB_map = {"Quechua":  0.09374108049701593,
                     "Aimara": 0.06945107405898711,
                     "Nativo o indígena de la amazonia": 0.06659724517518936, 
                     "Perteneciente o parte de otro pueblo":0.055086659208285385 ,
                     "Negro/moreno/zambo/mulato/pueblo afroperuano o afrodescendiente": 0.10598629350156058,
                     "Blanco":0.0888938831014238,
                    "Mestizo":0.10665882026914306,
                    "Otro":0.11565200806288764,
                    "No sabe":0.10641571219187955                      
                     }


CSALUD01_QS25BB = CSALUD01_QS25BB_map[CSALUD01_QS25BB_label]










education = st.selectbox(
    "Escolaridade",
    ["Primary", "Secondary", "High School", "University", "Postgraduate"])



# Quando clicar no botão, monta o dataframe e faz o score
if st.button("Calcular risco de parto prematuro"):
    # Monte o DF com os MESMOS nomes de colunas usados no treino
    input_data = pd.DataFrame([{
        "REC94_S410B": REC94_S410B,
        "REC41_M44": REC41_M44,
        "REC41_M14": REC41_M14,
        "education": education
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

    st.header("Dataframe")
    st.write(input_data)
