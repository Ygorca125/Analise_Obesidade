# Análise de Obesidade 🧘‍♂️📊

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

## 📋 Sobre o Projeto

O projeto foi desenvolvido com o objetivo de auxiliar uma equipe médica na previsão de níveis de obesidade. Além disso, foi criado um dashboard no Power BI para facilitar a análise da base de dados e apoiar a tomada de decisões estratégicas.

**Objetivo:** Modelo com assertividade preditiva acima de 75% utilizando streamlit.

---

## 🛠️ Tecnologias Utilizadas

- **Python** (EDA e Machine Learning)
- **Google Colab** / Jupyter Notebook
- **Streamlit** (app - front end)
- **Power BI** (visualizações)
- **Docker** (conteinerização)
- **Render** (hospedagem)

---

## 📂 Arquivos do Projeto

- `Obesity.csv` → Base bruta em csv
- `Analise_Obesidade.ipynb` → Notebook principal com toda a análise (EDA + Modelo Machine Learning)
- `class_mapping.pkl` → Converte a previsão numérica em texto legível (ex: 3 → "Obesidade Tipo III")
- `preprocessor.pkl` → Transforma os dados que o usuário digita no mesmo formato que o modelo foi treinado
- `xgboost_model.pkl` → Faz a previsão de nível de obesidade
- `requirements.txt` → Lista de todas as bibliotecas (pacotes) que o seu projeto precisa para rodar corretamente.
- `app.py` → Rodar a interface do Streamlit
- `Dockerfile` → Arquivo necessário para utilizar docker
- `PowerBI_Traducao_.ipynb` → Tradução dos dados para utilizar no PowerBi
- `Obesity_Traduzido_PowerBI.csv` → CSV do arquivo traduzido
- `Análise BI.pbix` → Arquivo em PowerBI
- `Análise BI.pdf` → PowerBI no formato PDF
- `Apresentação Projeto Módulo 4.pptx` → Apresentação final do projeto

---

## 🚀 Como Executar

1. Abra o Link do Streamlit e aguarde alguns segundos https://analise-obesidade.onrender.com/
2. Adicione no visual as informações e depois rode a previsão! 

