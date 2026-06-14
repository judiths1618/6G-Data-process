FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY dashboard/requirements.txt /tmp/dashboard-requirements.txt
RUN pip install --no-cache-dir -r /tmp/dashboard-requirements.txt

EXPOSE 8502

CMD ["streamlit", "run", "/app/dashboard/app.py", "--server.port=8502", "--server.address=0.0.0.0", "--server.headless=true"]
