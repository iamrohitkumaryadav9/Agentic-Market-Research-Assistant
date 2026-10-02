FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Create run_traces directory
RUN mkdir -p run_traces

# FastAPI (8000) + Streamlit (8501)
EXPOSE 8000 8501

ENV API_BASE=http://localhost:8000

# Run both services; if either exits the container stops (so a crash is visible).
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & \
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true & \
wait -n; exit $?"]
