# Usando imagem oficial do Python
FROM python:3.11-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos necessários
COPY requirements.txt .
COPY app.py .
COPY *.pkl ./
COPY Obesity.csv ./

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta 8501 (padrão do Streamlit)
EXPOSE 8501

# Comando para rodar o app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]