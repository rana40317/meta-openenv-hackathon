FROM python:3.11-slim
 
WORKDIR /app
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY env/ ./env/
COPY graders/ ./graders/
COPY server/ ./server/
COPY app.py .
COPY openenv.yaml .
COPY inference.py .
 
RUN touch env/__init__.py graders/__init__.py server/__init__.py
 
EXPOSE 7860
 
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
