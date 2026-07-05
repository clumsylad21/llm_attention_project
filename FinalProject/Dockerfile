FROM python:3.12-slim

WORKDIR /app

COPY FinalProject/requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn pydantic

COPY FinalProject/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]