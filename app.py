import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Preditor de Obesidade", layout="centered", page_icon="🎯")

st.title("🎯 Preditor de Nível de Obesidade")
st.markdown("Modelo XGBoost treinado")

# ==================== CARREGAR MODELO ====================
@st.cache_resource
def carregar_modelo():
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("class_mapping.pkl", "rb") as f:
        class_mapping = pickle.load(f)
    st.success("✅ Modelo XGBoost carregado!")
    return model, class_mapping

model, class_mapping = carregar_modelo()

# ==================== PREPROCESSOR ====================
@st.cache_resource
def criar_preprocessor():
    df = pd.read_csv("Obesity.csv")
    df_features = df.drop(columns=['Obesity'])

    categorical_cols = ['Gender', 'family_history', 'FAVC', 'CAEC', 
                        'SMOKE', 'SCC', 'CALC', 'MTRANS']
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_cols)
        ],
        remainder='drop'
    )
    
    preprocessor.fit(df_features)
    return preprocessor

preprocessor = criar_preprocessor()
st.success("✅ Preprocessador criado!")

# ==================== INPUTS EM PORTUGUÊS ====================
st.sidebar.header("📋 Dados da Pessoa")

# Gênero
genero_pt = st.sidebar.selectbox("Gênero", ["Feminino", "Masculino"])
gender = "Female" if genero_pt == "Feminino" else "Male"

age = st.sidebar.number_input("Idade (anos)", 10, 100, 25)
height = st.sidebar.number_input("Altura (m)", 1.0, 2.5, 1.70, 0.01)
weight = st.sidebar.number_input("Peso (kg)", 30, 200, 70)

# Histórico familiar
hist_fam_pt = st.sidebar.selectbox("Histórico familiar de obesidade", ["Não", "Sim"])
family_history = "yes" if hist_fam_pt == "Sim" else "no"

# FAVC
favc_pt = st.sidebar.selectbox("Consumo frequente de alimentos calóricos", ["Não", "Sim"])
favc = "yes" if favc_pt == "Sim" else "no"

# FCVC
fcvc = st.sidebar.slider("Frequência de consumo de vegetais", 1.0, 3.0, 2.0, 0.1)

# NCP
ncp = st.sidebar.slider("Número de refeições principais por dia", 1.0, 4.0, 3.0, 0.1)

# CAEC
caec_pt = st.sidebar.selectbox("Consumo de comida entre refeições", 
                               ["Nunca", "Às vezes", "Frequentemente", "Sempre"])
caec_map = {"Nunca": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
caec = caec_map[caec_pt]

# SMOKE
smoke_pt = st.sidebar.selectbox("Fuma?", ["Não", "Sim"])
smoke = "yes" if smoke_pt == "Sim" else "no"

# CH2O
ch2o = st.sidebar.slider("Consumo diário de água", 1.0, 3.0, 2.0, 0.1)

# SCC
scc_pt = st.sidebar.selectbox("Monitora ingestão calórica?", ["Não", "Sim"])
scc = "yes" if scc_pt == "Sim" else "no"

# FAF
faf = st.sidebar.slider("Frequência de atividade física semanal", 0.0, 3.0, 1.0, 0.1)

# TUE
tue = st.sidebar.slider("Tempo diário usando dispositivos eletrônicos", 0.0, 2.0, 1.0, 0.1)

# CALC
calc_pt = st.sidebar.selectbox("Consumo de álcool", 
                               ["Nunca", "Às vezes", "Frequentemente", "Sempre"])
calc_map = {"Nunca": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
calc = calc_map[calc_pt]

# MTRANS
mtrans_pt = st.sidebar.selectbox("Meio de transporte habitual", 
    ["Transporte Público", "Carro", "Moto", "Bicicleta", "A pé"])
mtrans_map = {
    "Transporte Público": "Public_Transportation",
    "Carro": "Automobile",
    "Moto": "Motorbike",
    "Bicicleta": "Bike",
    "A pé": "Walking"
}
mtrans = mtrans_map[mtrans_pt]

# ==================== PREVISÃO ====================
if st.sidebar.button("🔮 Fazer Previsão", type="primary", use_container_width=True):
    with st.spinner("Calculando previsão..."):
        dados = {
            "Gender": [gender],
            "Age": [age],
            "Height": [height],
            "Weight": [weight],
            "family_history": [family_history],
            "FAVC": [favc],
            "FCVC": [fcvc],
            "NCP": [ncp],
            "CAEC": [caec],
            "SMOKE": [smoke],
            "CH2O": [ch2o],
            "SCC": [scc],
            "FAF": [faf],
            "TUE": [tue],
            "CALC": [calc],
            "MTRANS": [mtrans]
        }
        
        input_df = pd.DataFrame(dados)
        input_processed = preprocessor.transform(input_df)
        
        # Remove feature extra se necessário
        if input_processed.shape[1] == 23:
            input_processed = input_processed[:, :22]
        
        pred = model.predict(input_processed)[0]
        classe = class_mapping.get(pred, pred) if isinstance(class_mapping, dict) else pred
        
        st.success(f"**Nível de Obesidade Previsto: {classe}**")

st.caption("App desenvolvido com Streamlit + XGBoost")