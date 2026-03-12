FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Ollama (CPU)
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Python deps
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Streamlit / networking
ENV PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=$PORT
ENV OLLAMA_HOST=0.0.0.0

EXPOSE 11434
EXPOSE 7860

# Start Ollama daemon, pull model on boot, then Streamlit
CMD ["bash","-lc","(ollama serve &) && echo 'Waiting for Ollama…' && sleep 6 && ollama pull ${OLLAMA_MODEL:-llama3.2:3b-instruct-q4_K_M} || true; streamlit run app.py --server.address 0.0.0.0 --server.port $PORT"]

