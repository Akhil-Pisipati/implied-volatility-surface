FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY core ./core
COPY data ./data
COPY api ./api
COPY docs ./docs
COPY app.py ./app.py
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir .
EXPOSE 8501 8000
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501"]
