FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Download and cache the model during build
RUN python -c "from transformers import AutoTokenizer, AutoModelForMaskedLM; \
    model_name='tahrirchi/tahrirchi-bert-base'; \
    AutoTokenizer.from_pretrained(model_name); \
    AutoModelForMaskedLM.from_pretrained(model_name)"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]