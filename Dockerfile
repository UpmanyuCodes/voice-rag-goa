FROM python:3.11-slim

LABEL maintainer="HH Goa 2026 Team"
LABEL description="Voice-Enabled RAG System — HH Goa 2026 Task 2"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY web/ web/

# Environment defaults
ENV PORT=8000
ENV HOST=0.0.0.0
ENV STT_PROVIDER=sarvam
ENV LLM_PROVIDER=groq
ENV ENABLE_SEMANTIC_CACHE=true
ENV LATENCY_TARGET_MS=200

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
