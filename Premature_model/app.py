import streamlit as st
import pandas as pd
import pickle
import os
import sys

# --- Load model ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "trained_model_4_20251212_202758.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()


# --- UI ---
st.title("PremaTerm Risk Score")
st.write("Aplicación de predicción inteligente de riesgo de parto prematuro")

st.markdown("""
### Instrucciones para el profesional de salud

Complete cada campo con la información clínica actual de la paciente.

Este instrumento de apoyo diagnóstico emplea un modelo de aprendizaje automático.  
El sistema procesará los datos de forma inmediata, entregará un porcentaje de riesgo estimado y la predicción del paciente.

**★ Todos los campos son obligatorios para obtener una predicción válida.**
""")


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


# REC91_V228
REC91_V228_label = st.selectbox("¿Ha tenido usted alguna vez un embarazo que terminara en pérdida, aborto o nacido muerto?",
                         ["Si", "No"])
REC91_V228_map = {"Si":  1,"No": 0}
REC91_V228 = REC91_V228_map[REC91_V228_label]

# RE516171_V717
RE516171_V717_label = st.selectbox("¿Tiene un trabajo o negocio del cual estuvo ausente por licencia, enfermedad, vacaciones, maternidad o cualquier otra razón?:",
                         ["No trabaja", 
                        "Profesional, Técnico, Gerente",
                        "Eclesiástico",
                        "Ventas",
                        "Agricultor, trabajador independiente",
                        "Agricultor, empleado", 
                        "Empleada del hogar",
                        "Servicios",
                        "Habilidades manuales",
                        "Sin habilidades manuales"                        
                        ])
RE516171_V717_map = {
                        "No trabaja":0.08393339445223347, 
                        "Profesional, Técnico, Gerente":0.14884821825675165,
                        "Eclesiástico": 0.11758076629505221,
                        "Ventas":0.11817795517255232,
                        "Agricultor, trabajador independiente":0.0859559533163348,
                        "Agricultor, empleado": 0.0859559533163348,
                        "Empleada del hogar":0.10018800431227122,
                        "Servicios":0.125842708558033,
                        "Habilidades manuales":0.10973043092481712,
                        "Sin habilidades manuales": 0.10636902121916186,                       
                     }
        

RE516171_V717 = RE516171_V717_map[RE516171_V717_label]


# QS25AA - Idioma o lengua materna aprendida en la niñez
CSALUD01_QS25AA_label = st.selectbox(
    "¿Cuál es el idioma o lengua materna que aprendió hablar en su niñez?",
    [
        "Quechua",
        "Aimara",
        "Ashaninka",
        "Awajun/Aguaruna",
        "Shipibo/Konibo",
        "Shawi/Chayahuita",
        "Matsigenka/Machiguenga",
        "Achuar",
        "Otra lengua nativa u originaria",
        "Castellano",
        "Portugués",
        "Otra lengua extranjera"
    ]
)

# --- Mapping according to the values you provided ---
CSALUD01_QS25AA_map = {
    "Quechua": 0.07923371838921675,
    "Aimara": 0.05442813420625482,
    "Ashaninka": 0.07874633165084714,
    "Awajun/Aguaruna": 0.023129927769073508,
    "Shipibo/Konibo": 0.03561381880949324,
    "Shawi/Chayahuita": 0.16457011794224158,
    "Matsigenka/Machiguenga": 0.05625986263317399,
    "Achuar": 0.1232005043368985,
    "Otra lengua nativa u originaria": 0.031822232944330515,
    "Castellano": 0.10727716728563563,
    "Portugués": 0.10848680431710521,
    "Otra lengua extranjera": 0.21883524418464348
}

CSALUD01_QS25AA = CSALUD01_QS25AA_map[CSALUD01_QS25AA_label]


# REC41_M45
REC41_M45_label = st.selectbox("Durante el embarazo, ¿tomó hierro?",
                         ["Si", "No", "No sabe"])
REC41_M45_map = {"Si":  0.19239657978375568,"No": 0.19424243862086038,"No sabe": 0.6209869634191607}
REC41_M45 = REC41_M45_map[REC41_M45_label]



#RE516171_V501

# --- RE516171_V501: Estado civil actual ---
RE516171_V501_label = st.selectbox(
    "¿Cuál es su estado civil actual?",
    [
        "Nunca casada",
        "Casado",
        "Viviendo juntos",
        "Viuda",
        "Divorciada",
        "No viven juntos"
    ]
)

RE516171_V501_map = {
    "Nunca casada": 0.018683965078066432,
    "Casado": 0.15580028078089597,
    "Viviendo juntos": 0.13813044249912082,
    "Viuda": 0.04034764914742743,
    "Divorciada": 0.0845248941723454,
    "No viven juntos": 0.12137826887416875
}

# Valor final para mandar ao modelo
RE516171_V501 = RE516171_V501_map[RE516171_V501_label]


REC94_S411BA= st.number_input("¿Cuantos meses de embarazo tenía usted cuando le realizaron el primer Examen de Orina?", 
                              min_value=1, max_value=50, value=1)



# --- CSALUD01_QS27: Seguro de salud actual ---

CSALUD01_QS27_label = st.selectbox(
    "¿Qué tipo de seguro de salud tiene actualmente?",
    [
        "Seguro Integral de Salud",
        "ESSALUD / IPSS",
        "Fuerzas Armadas o Policiales",
        "Entidad Prestadora de Salud (EPS)",
        "Seguro Privado",
        "Otro"
    ]
)

CSALUD01_QS27_map = {
    "Seguro Integral de Salud": 0.09816262732470862,  # A
    "ESSALUD / IPSS": 0.12136395263522679,            # B
    "Fuerzas Armadas o Policiales": 0.126977005275932, # C
    "Entidad Prestadora de Salud (EPS)": 0.21405169393148776, # D
    "Seguro Privado": 0.20496226075255936,            # E
    "Otro": 0.0                                     # X

}

CSALUD01_QS27 = CSALUD01_QS27_map[CSALUD01_QS27_label]



# RE516171_V704

RE516171_V704_label = st.selectbox(
    "¿Cuál es su ocupación actual?",
    [
        "Otro, no trabaja, no especificada",     # code 0
        "Fuerzas Armadas",                       # code 1
        "Fuerzas Policiales",                    # code 2
        "Pensionado, Jubilado, Rentista, Estudiante",  # code 5
        "Miembros del Poder Ejecutivo y Cuerpos Legislativos",  # 11
        "Directores de Empresa (3 o más Directores)",          # 12
        "Directores de la Educación",                          # 13
        "Gerente de Pequeñas Empresas (Talleres)",            # 14
        "Profesionales de Ciencias Físicas, Químicas, Matemáticas, Ingeniería, Arquitectura",  # 21
        "Ingenierías (excepto Ing. Civil)",                    # 22
        "Profesionales de Ciencias Biológicas, Medicina y Salud",  # 23
        "Profesores, Maestros, Pedagogos",                     # 24
        "Profesionales de Derecho y Ciencias Económicas",      # 25
        "Bibliotecario, Periodistas, Psicólogos, Ciencias Sociales", # 26
        "Profesionales de Artes, Música, Escultura",           # 27
        "Profesionales de Turismo, Hotelería, Diplomáticos, Sacerdotes", # 28
        "Técnicos en Ciencias Físicas, Químicas, Matemáticas e Ingeniería", # 31
        "Operadores de Equipos Especializados",                # 32
        "Técnicos en Navegación, Inspectores de Calidad/Seguridad", # 33
        "Técnicos en Ciencias Biológicas",                     # 34
        "Técnicos en Medicina y Salud",                        # 35
        "Jefes de Ventas, Técnicos en Administración y Finanzas", # 36
        "Agentes de Servicios Administrativos",                # 37
        "Asistentes y Auxiliares Administrativos",             # 38
        "Artistas, Atletas, Espectáculo, Auxiliares de Culto", # 39
        "Jefes Administrativos, Secretarias, Digitadores",     # 41
        "Personal Administrativo y Afines",                    # 42
        "Jefes de Servicios de Transporte",                    # 43
        "Bibliotecas, Imprenta, Correos, Mensajeros",          # 44
        "Cajeros, Recepcionistas, Telefonistas",               # 45
        "Encuestadores, Registradores, Oficinistas de Campo", # 46
        "Guías de Turismo, Aeromozas",                         # 51
        "Jefes de Servicio, Mozo, Azafata",                    # 52
        "Auxiliares de Enfermería y Cuidados Personales",      # 53
        "Peluqueros y Tratamiento de Belleza",                 # 54
        "Servicios Varios, Modelos, Tramitadores",             # 55
        "Bomberos, Serenazgo, Seguridad",                      # 56
        "Comerciantes y Vendedores Minoristas",                # 57
        "Vendedores no Ambulatorios",                          # 58
        "Agricultores y Trabajadores Agropecuarios",           # 61
        "Criadores y Trabajadores Pecuarios",                  # 62
        "Pescadores, Cazadores, Tramperos",                    # 63
        "Mineros, Canteros",                                   # 71
        "Tratamiento de la Madera y Papel",                    # 72
        "Tratamientos Químicos",                               # 73
        "Hilanderos y Tintoreros",                             # 74
        "Tratamiento de Pieles",                               # 75
        "Preparación de Alimentos (Carnicero, Panadero)",      # 76
        "Calzado, Sastres, Ebanistas",                         # 77
        "Mecánicos y Ajustadores de Vehículos",                # 78
        "Ajustadores e Instaladores de Máquinas",              # 79
        "Plástico, Música, Pintura Industrial",                # 81
        "Productos de Papel y Cartón",                         # 82
        "Artes Gráficas, Fotografía",                          # 83
        "Manufactura No Clasificada",                          # 84
        "Pintores de Edificios y Decoración",                  # 85
        "Construcción: Albañil, Carpintero",                   # 86
        "Operadores de Maquinaria Industrial",                 # 87
        "Conductores de Transporte",                           # 88
        "Vendedores Ambulantes",                               # 91
        "Cobradores de Transporte",                            # 93
        "Servicio Doméstico",                                  # 94
        "Mensajeros, Repartidores, Porteros",                  # 95
        "Recolectores de Basura",                              # 96
        "Peones de Minería y Energía"                      # 98

    ]
)

RE516171_V704_map = {
    "Otro, no trabaja, no especificada": 0.018683965078066432,   # code 0
    "Fuerzas Armadas": 0.22542236341301503,                      # code 1
    "Fuerzas Policiales": 0.15085775938463078,                   # code 2
    "Pensionado, Jubilado, Rentista, Estudiante": 0.11066693577566691, # code 5
    "Miembros del Poder Ejecutivo y Cuerpos Legislativos": 0.08461058556915746, # 11
    "Directores de Empresa (3 o más Directores)": 1.0,           # 12
    "Directores de la Educación": 0.21436145275554663,           # 13
    "Gerente de Pequeñas Empresas (Talleres)": 0.12653982438017142, # 14
    "Profesionales de Ciencias Físicas, Químicas, Matemáticas, Ingeniería, Arquitectura": 0.1634465901043226, # 21
    "Ingenierías (excepto Ing. Civil)": 0.2226361995769824,      # 22
    "Profesionales de Ciencias Biológicas, Medicina y Salud": 0.17951293061473358, # 23
    "Profesores, Maestros, Pedagogos": 0.15477967916970056,      # 24
    "Profesionales de Derecho y Ciencias Económicas": 0.19002422371995406, # 25
    "Bibliotecario, Periodistas, Psicólogos, Ciencias Sociales": 0.19146702141806976, # 26
    "Profesionales de Artes, Música, Escultura": 0.1539618720183686, # 27
    "Profesionales de Turismo, Hotelería, Diplomáticos, Sacerdotes": 0.24507539001949386, # 28
    "Técnicos en Ciencias Físicas, Químicas, Matemáticas e Ingeniería": 0.18684599452622794, # 31
    "Operadores de Equipos Especializados": 0.04888613600204209, # 32
    "Técnicos en Navegación, Inspectores de Calidad/Seguridad": 0.20416044979090683, # 33
    "Técnicos en Ciencias Biológicas": 0.17453161337126782,      # 34
    "Técnicos en Medicina y Salud": 0.10828885784836847,         # 35
    "Jefes de Ventas, Técnicos en Administración y Finanzas": 0.18138620135840638, # 36
    "Agentes de Servicios Administrativos": 0.26218598155759604, # 37
    "Asistentes y Auxiliares Administrativos": 0.11104445939981385, # 38
    "Artistas, Atletas, Espectáculo, Auxiliares de Culto": 0.1538795570530606, # 39
    "Jefes Administrativos, Secretarias, Digitadores": 0.17908842541816106, # 41
    "Personal Administrativo y Afines": 0.15847680664397878,     # 42
    "Jefes de Servicios de Transporte": 0.0,                     # 43
    "Bibliotecas, Imprenta, Correos, Mensajeros": 0.06823316681697436, # 44
    "Cajeros, Recepcionistas, Telefonistas": 0.12260856992144328, # 45
    "Encuestadores, Registradores, Oficinistas de Campo": 0.17951741920343162, # 46
    "Guías de Turismo, Aeromozas": 0.1051570550834244,           # 51
    "Jefes de Servicio, Mozo, Azafata": 0.12600364598878896,     # 52
    "Auxiliares de Enfermería y Cuidados Personales": 0.0,       # 53
    "Peluqueros y Tratamiento de Belleza": 0.23298485471464905,  # 54
    "Servicios Varios, Modelos, Tramitadores": 0.0,              # 55
    "Bomberos, Serenazgo, Seguridad": 0.1736499437797703,        # 56
    "Comerciantes y Vendedores Minoristas": 0.1422583662431103,  # 57
    "Vendedores no Ambulatorios": 0.13654829995674425,           # 58
    "Agricultores y Trabajadores Agropecuarios": 0.04977900905466541, # 61
    "Criadores y Trabajadores Pecuarios": 0.12561842732238218,   # 62
    "Pescadores, Cazadores, Tramperos": 0.12320753836251644,     # 63
    "Mineros, Canteros": 0.10740182656901472,                    # 71
    "Tratamiento de la Madera y Papel": 0.0,                     # 72
    "Tratamientos Químicos": 0.0,                                # 73
    "Hilanderos y Tintoreros": 0.048886136002042085,             # 74
    "Tratamiento de Pieles": 0.0,                                # 75
    "Preparación de Alimentos (Carnicero, Panadero)": 0.1588549479493787, # 76
    "Calzado, Sastres, Ebanistas": 0.15015535352237827,          # 77
    "Mecánicos y Ajustadores de Vehículos": 0.16599384371199505, # 78
    "Ajustadores e Instaladores de Máquinas": 0.13640474432942543, # 79
    "Plástico, Música, Pintura Industrial": 0.10520872254782203, # 81
    "Productos de Papel y Cartón": 0.0,                          # 82
    "Artes Gráficas, Fotografía": 0.3189083758378702,            # 83
    "Manufactura No Clasificada": 0.1702717935454207,            # 84
    "Pintores de Edificios y Decoración": 0.07711537108405837,  # 85
    "Construcción: Albañil, Carpintero": 0.13025232151918378,   # 86
    "Operadores de Maquinaria Industrial": 0.14237198234066373, # 87
    "Conductores de Transporte": 0.15251751412134576,           # 88
    "Vendedores Ambulantes": 0.1093244352689116,                # 91
    "Cobradores de Transporte": 0.20072090325993222,            # 93
    "Servicio Doméstico": 0.15231251370037485,                  # 94
    "Mensajeros, Repartidores, Porteros": 0.13619000876055232,  # 95
    "Recolectores de Basura": 0.14466961711347132,              # 96
    "Peones de Minería y Energía": 0.14233287548720694,         # 98
}

RE516171_V704 = RE516171_V704_map[RE516171_V704_label]


# REC41_M13

REC41_M13= st.number_input("¿Cuántos meses de embarazo tenía Ud. cuando se hizo su primer control prenatal?", 
                              min_value=1, max_value=50, value=1)



#REC41_M43
REC41_M43_label = st.selectbox("¿Le explicaron acerca de las complicaciones que se pueden presentar en el embarazo?",
                         ["Si", "No", "No sabe"])
REC41_M43_map = {"Si":  0.19004575334267299,"No": 0.24648950320607282,"No sabe": 0.351137384963986}
REC41_M43 = REC41_M43_map[REC41_M43_label]



#RE516171_V743F


RE516171_V743F_label = st.selectbox(
    "¿Quién decide principalmente cómo se gasta el dinero que su esposo/compañero gana?",
    [
        "Entrevistada",
        "Entrevistada y esposo/compañero",
        "Entrevistada y alguien más",
        "Esposo/compañero",
        "Alguien más",
        #"Otro",
        "Esposo/compañero no tiene ganancias"
    ]
)

RE516171_V743F_map = {
    "Entrevistada": 0.16303102800894273,                     # 1
    "Entrevistada y esposo/compañero": 0.0881069176215889,   # 2
    "Entrevistada y alguien más": 0.0,                       # 3
    "Esposo/compañero": 0.13342486748964463,                 # 4
    "Alguien más": 0.13654829995674425,                      # 5
    #"Otro": 0.0,                                             # 6 (no value in map → safe fallback)
    "Esposo/compañero no tiene ganancias": 0.1364563460702525 # 7
}

RE516171_V743F = RE516171_V743F_map[RE516171_V743F_label]


# REC41_M48
REC41_M48_label = st.selectbox("¿Durante el embarazo ¿Tenía usted algún problema para ver los objetos, cosas o personas siendo de noche?",
                         ["Si", "No", "No sabe"])
REC41_M48_map = {"Si":  0.25560398731401435,"No": 0.19059454787050883,"No sabe": 0.0}
REC41_M48 = REC41_M48_map[REC41_M48_label]

# RE516171_V715
RE516171_V715= st.number_input("Educación en años individuales del esposo/compañero",
                                min_value=1, max_value=100, value=1)



# REC91_V217

REC91_V217 = st.number_input(
    """¿Cree usted que hay ciertos días en los cuales una mujer,si tiene relaciones sexuales,
      puede quedar más fácilmente embarazada? Digite el numero al cual corresponde: 1-Durante su periodo, 2-Después de su período,
      3-En medio del ciclo, 4-Antes del período, 5-En cualquier momento,6-Otro, 8-No sabe      
      """,
    min_value=1,
    max_value=50,
    value=1
    )


#REC91_V218
REC91_V218= st.number_input("¿Cuántos niños nacidos vivos tiene?",
                                min_value=1, max_value=100, value=1)


#REC94_S413
REC94_S413_label = st.selectbox("¿Durante el embarazo  Ud. estaba afiliada al Seguro Integral de Salud o Materno-infantil?",
                         ["Si", "No"])
REC94_S413_map = {"Si":  1,"No": 0}
REC94_S413 = REC94_S413_map[REC94_S413_label]

#REC41_M57K
REC41_M57K_label = st.selectbox("¿Se controlo  su ultimo hijo (a) nacido en Policlínico/centro/posta ESSALUD ?",
                         ["Si", "No"])
REC41_M57K_map = {"Si":  1,"No": 0}
REC41_M57K = REC41_M57K_map[REC41_M57K_label]


#CSALUD01_QS102
CSALUD01_QS102_label = st.selectbox("¿Alguna vez en su vida un médico le ha diagnosticado hipertensión arterial o presión alta?",
                         ["Si", "No", "No sabe"])
CSALUD01_QS102_map = {"Si":  0.08071005999283234,"No": 0.10124067311025016,"No sabe": 0.09998061589271888}
CSALUD01_QS102 = CSALUD01_QS102_map[CSALUD01_QS102_label]


#REC94_S411H
REC94_S411H_label = st.selectbox("¿En alguno de sus controles le hicieron la prueba para descartar VIH/SIDA?",
                         ["Si", "No", "No sabe"])
REC94_S411H_map = {"Si":  0.1924953544634142,"No": 0.20440853439084816,"No sabe": 0.16523836902803687}
REC94_S411H = REC94_S411H_map[REC94_S411H_label]

#REC41_M57M
REC41_M57M_label = st.selectbox("¿Se controlo  su ultimo hijo (a) nacido en Hospital privado/clínica?",
                         ["Si", "No"])
REC41_M57M_map = {"Si":  1,"No": 0}
REC41_M57M = REC41_M57M_map[REC41_M57M_label]


#REC94_S411G
REC94_S411G_label = st.selectbox("¿Durante su embarazo le hicieron la prueba para descartar Sífilis?",
                         ["Si", "No", "No sabe"])
REC94_S411G_map = {"Si":  0.191715061553752,"No": 0.21346031315512892,"No sabe": 0.17641407166540363}
REC94_S411G = REC94_S411G_map[REC94_S411G_label]


#REC41_M60
REC41_M60_label = st.selectbox("¿Durante el embarazo ¿tomó algún medicamento contra las lombrices o los gusanos intestinales?",
                         ["Si", "No", "No sabe"])
REC41_M60_map = {"Si":  0.20775713209080082,"No": 0.19214411518477217,"No sabe": 0.09188315574750562}
REC41_M60 = REC41_M60_map[REC41_M60_label]


#RE516171_V705

RE516171_V705_label = st.selectbox(
    "¿Cuál es la ocupación de su esposo/compañero?",
    [
        "No trabaja",                       # 0
        "Profesional, Técnico, Gerente",    # 1
        "Eclesiástico",                     # 2
        "Ventas",                           # 3
        "Agricultor, trabajador independiente",  # 4
        "Agricultor, empleado",             # 5 (not in map → assigned same as 4)
        "Empleado del hogar",               # 6
        "Servicios",                        # 7
        "Habilidades manuales",             # 8
        "Sin habilidades manuales"          # 9
    ]
)

RE516171_V705_map = {
    "No trabaja": 0.13717770789276298,                      # 0
    "Profesional, Técnico, Gerente": 0.1757786992527622,   # 1
    "Eclesiástico": 0.16025676487233698,                   # 2
    "Ventas": 0.14105326585085395,                         # 3
    "Agricultor, trabajador independiente": 0.05177079502952714, # 4
    "Agricultor, empleado": 0.05177079502952714,           # 5 (fallback)
    "Empleado del hogar": 0.14671666184378934,             # 6
    "Servicios": 0.15601869410139355,                      # 7
    "Habilidades manuales": 0.14300093887130894,           # 8
    "Sin habilidades manuales": 0.14244410262792964        # 9
}

RE516171_V705= RE516171_V705_map[RE516171_V705_label]


#REC94_S411F
REC94_S411F_label = st.selectbox("¿Escucharon los latidos del corazón del bebé en alguno de sus controles?",
                         ["Si", "No", "No sabe"])
REC94_S411F_map = {"Si":  0.1921804689297172,"No": 0.2540690771492045,"No sabe": 0.17004408699980744}
REC94_S411F = REC94_S411F_map[REC94_S411F_label]

#REC94_QI422A_A
REC94_QI422A_A_label = st.selectbox("Durante el embarazo ¿Algún personal de salud le realizó una prueba o análisis para descartar anemia?",
                         ["Si", "No", "No sabe"])
REC94_QI422A_A_map = {"Si":  0.19107895270772216,"No": 0.24261635765951195,"No sabe": 0.15085811217129316}
REC94_QI422A_A = REC94_QI422A_A_map[REC94_QI422A_A_label]

#CSALUD01_QS107
CSALUD01_QS107_label = st.selectbox("Durante en embarazo ¿Alguna vez le han medido la glucosa o el azucar en la sangre?",
                         ["Si", "No", "No sabe"])
CSALUD01_QS107_map = {"Si":  0.10292754143877926,"No": 0.09831823934032337,"No sabe": 0.15057928881139188}
CSALUD01_QS107 = CSALUD01_QS107_map[CSALUD01_QS107_label]

#RE516171_V743E

RE516171_V743E_label = st.selectbox(
    "¿Quién decide sobre qué comida se debe cocinar cada día?",
    [
        "Nadie",                        # 0
        "Entrevistada",                 # 1
        "Entrevistada y esposo/compañero",  # 2
        "Entrevistada y alguien más",   # 3
        "Esposo/compañero",             # 4
        "Alguien más"                   # 5
    ]
)

RE516171_V743E_map = {
    "Nadie": 0.1001473142056087,                        # 0
    "Entrevistada": 0.09355102620073219,                # 1
    "Entrevistada y esposo/compañero": 0.16252000516341472,  # 2
    "Entrevistada y alguien más": 0.11161388753945603,  # 3
    "Esposo/compañero": 0.14486037760713413,            # 4
    "Alguien más": 0.18433560495130968                  # 5
}

RE516171_V743E = RE516171_V743E_map[RE516171_V743E_label]


#REC94_S441

REC94_S441_label = st.selectbox("¿Recibió alguna capacitación (charla, enseñanza) sobre lactancia materna durante su embarazo?",
                         ["Si", "No"])
REC94_S441_map = {"Si":  1,"No": 0}
REC94_S441 = REC94_S441_map[REC94_S441_label]



#RE516171_V732
RE516171_V732_label = st.selectbox(
    "¿Ud. usualmente trabaja (trabajaba) durante todo el año, trabaja por temporada o sólo de vez en cuando?",
    [
        "Todo el año",      # 1
        "Por temporada",    # 2
        "De vez en cuando"  # 3
    ]
)

RE516171_V732_map = {
    "Todo el año": 0.09866252651072997,    # 1
    "Por temporada": 0.114104740380939,    # 2
    "De vez en cuando": 0.1135892274315984 # 3
}

RE516171_V732 = RE516171_V732_map[RE516171_V732_label]


#REC41_M2C
REC41_M2C_label = st.selectbox("¿La chequeó en su control prenatal un obstetra a su ultimo hijo",
                         ["Si", "No"])
REC41_M2C_map = {"Si":  1,"No": 0}
REC41_M2C = REC41_M2C_map[REC41_M2C_label]

#CSALUD01_QS207C

CSALUD01_QS207C= st.number_input(" ¿Qué edad tenía usted la primera vez que tomó alguna bebida alcohólica o licor?",
                                min_value=1, max_value=100, value=1)


#REC94_S411DA
REC94_S411DA= st.number_input("¿Cuantos meses de embarazo tenía usted cuando le realizaron la primera prueba para descartar Sífiis?",
                                min_value=1, max_value=100, value=1)

#REC41_M42E
REC41_M42E_label = st.selectbox("Durante su embarazo en alguno de sus controles ¿le hicieron examen de sangre? ",
                         ["Si", "No", "No sabe"])
REC41_M42E_map = {"Si":  0.19185297587197733,"No": 0.2230984739996829,"No sabe": 0.2447751915073055}
REC41_M42E = REC41_M42E_map[REC41_M42E_label]




# Quando clicar no botão, monta o dataframe e faz o score
if st.button("Calcular risco de parto prematuro"):
    # Monte o DF com os MESMOS nomes de colunas usados no treino
    input_data = pd.DataFrame([{
    "REC94_S410B": REC94_S410B,
    "REC41_M44": REC41_M44,
    "REC41_M14": REC41_M14,
    "REC94_S411J": REC94_S411J,
    "REC94_QI422A_B": REC94_QI422A_B,
    "RE516171_V631": RE516171_V631,
    "REC94_S411K": REC94_S411K,
    "REC94_S411I": REC94_S411I,
    "RE516171_V504": RE516171_V504,
    "REC41_M42C": REC41_M42C,
    "REC91_V228": REC91_V228,
    "REC41_M43": REC41_M43,
    "REC41_M45": REC41_M45,
    "CSALUD01_QS25AA": CSALUD01_QS25AA,
    "REC91_V208": REC91_V208,
    "CSALUD01_QS27": CSALUD01_QS27,
    "REC91_V212": REC91_V212,
    "REC41_M13": REC41_M13,
    "CSALUD01_QS25BB": CSALUD01_QS25BB,
    "REC41_M48": REC41_M48,
    "RE516171_V717": RE516171_V717,
    "RE516171_V743A": RE516171_V743A,
    "REC94_S411BA": REC94_S411BA,
    "RE516171_V501": RE516171_V501,
    "RE516171_V704": RE516171_V704,
    "RE516171_V743F": RE516171_V743F,
    "REC94_S413": REC94_S413,
    "REC91_V217": REC91_V217,
    "REC91_V218": REC91_V218,
    "CSALUD01_QS207C": CSALUD01_QS207C,
    "REC41_M42E": REC41_M42E,
    "REC94_S411H": REC94_S411H,
    "RE516171_V715": RE516171_V715,
    "REC41_M2C": REC41_M2C,
    "REC41_M57K": REC41_M57K,
    "RE516171_V732": RE516171_V732,
    "REC94_QI422A_A": REC94_QI422A_A,
    "CSALUD01_QS102": CSALUD01_QS102,
    "REC41_M57M": REC41_M57M,
    "REC94_S411DA": REC94_S411DA,
    "REC94_S411G": REC94_S411G,
    "CSALUD01_QS107": CSALUD01_QS107,
    "REC41_M60": REC41_M60,
    "REC94_S411F": REC94_S411F,
    "RE516171_V705": RE516171_V705,
    "RE516171_V743E": RE516171_V743E,
    "REC94_S441": REC94_S441
    }], columns=model.get_booster().feature_names)


    prob = model.predict_proba(input_data)[0][1]   # probability of class 1
    # st.markdown(f"### Risco estimado: **{prob:.2%}**")

    # if prob >=0.5:
    #     st.warning("Usuaio tendra parto prematuro")

    # else:
    #     st.warning("Usuaio NO tendra parto prematuro")

    if prob >= 0.5:
        bg_color = "#F7E2D2"
        inner_color = "#F5C49F"
        icon = "⚠️"
        message = "Paciente tendrá un parto prematuro."
        recommendation = "Recomendación de seguimiento intensificado."

    else:
        bg_color = "#E3F3DD"
        inner_color = "#C4E8B8"
        icon = "✅"
        message = "Paciente NO tendrá un parto prematuro."
        recommendation = "Recomendación de continuar controles habituales."

st.markdown(
    f"""
    <div style="
        background-color:{bg_color};
        border-radius:28px;
        padding:32px;
        margin-top:24px;
        margin-bottom:24px;
        color:#111111;
    ">
        <div style="display:flex; align-items:center; gap:24px;">
            <div style="font-size:48px;">{icon}</div>

            <div>
                <div style="font-size:28px; font-weight:700; color:#111111;">
                    Riesgo estimado:
                </div>
                <div style="font-size:44px; font-weight:700; color:#111111;">
                    {prob:.2%}
                </div>
            </div>
        </div>

        <div style="
            background-color:{inner_color};
            border-radius:18px;
            padding:22px;
            margin-top:26px;
            color:#111111;
        ">
            <div style="font-size:26px; font-style:italic; color:#111111;">
                {message}
            </div>
            <div style="font-size:18px; margin-top:8px; color:#111111;">
                {recommendation}
            </div>
        </div>
    </div>

    <p style="font-size:16px; color:#111111;">
        ⚠️ Este resultado es una predicción generada por un modelo de inteligencia artificial y no reemplaza el criterio clínico del profesional de salud. Debe interpretarse en conjunto con el contexto clínico completo de la paciente.
    </p>
    """,
    unsafe_allow_html=True
)