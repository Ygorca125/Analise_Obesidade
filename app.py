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

st.success("✅ Preprocessor criado!")

# ==================== INPUTS ====================
st.sidebar.header("📋 Dados da Pessoa")

gender = st.sidebar.selectbox("Gênero", ["Female", "Male"])
age = st.sidebar.number_input("Idade (anos)", 10, 100, 25)
height = st.sidebar.number_input("Altura (m)", 1.0, 2.5, 1.70, 0.01)
weight = st.sidebar.number_input("Peso (kg)", 30, 200, 70)

family_history = st.sidebar.selectbox("Histórico familiar", ["no", "yes"])
favc = st.sidebar.selectbox("FAVC", ["no", "yes"])
fcvc = st.sidebar.slider("FCVC", 1.0, 3.0, 2.0, 0.1)
ncp = st.sidebar.slider("NCP", 1.0, 4.0, 3.0, 0.1)
caec = st.sidebar.selectbox("CAEC", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.sidebar.selectbox("SMOKE", ["no", "yes"])
ch2o = st.sidebar.slider("CH2O", 1.0, 3.0, 2.0, 0.1)
scc = st.sidebar.selectbox("SCC", ["no", "yes"])
faf = st.sidebar.slider("FAF", 0.0, 3.0, 1.0, 0.1)
tue = st.sidebar.slider("TUE", 0.0, 2.0, 1.0, 0.1)
calc = st.sidebar.selectbox("CALC", ["no", "Sometimes", "Frequently"])
mtrans = st.sidebar.selectbox("MTRANS", ["Public_Transportation", "Automobile", "Motorbike", "Bike", "Walking"])

# ==================== PREVISÃO ====================
if st.sidebar.button("🚀 Fazer Previsão", type="primary"):
    dados = {
        "Gender": [gender], "Age": [age], "Height": [height],
        "family_history": [family_history], "FAVC": [favc],
        "FCVC": [fcvc], "NCP": [ncp], "CAEC": [caec],
        "SMOKE": [smoke], "CH2O": [ch2o], "SCC": [scc],
        "FAF": [faf], "TUE": [tue], "CALC": [calc],
        "MTRANS": [mtrans], "Weight": [weight]
    }
    
    input_df = pd.DataFrame(dados)
    
    input_processed = preprocessor.transform(input_df)
    
    # FORÇAR 22 FEATURES (ajuste necessário)
    if input_processed.shape[1] == 23:
        input_processed = input_processed[:, :22]   # remove a feature extra
    
    st.info(f"✅ Features usadas na previsão: {input_processed.shape[1]}")
    
    pred = model.predict(input_processed)[0]
    
    classe = class_mapping.get(pred, pred) if isinstance(class_mapping, dict) else pred
    st.success(f"**Nível de Obesidade Previsto: {classe}**")

st.caption("App feito com ❤️")