FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/tmp/huggingface
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY rag_lab/ ./rag_lab/
EXPOSE 8000
USER 65532:65532
CMD ["uvicorn", "rag_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
