FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY artifacts ./artifacts

# Copy only the labels CSV needed for fallback vocab rebuild / metadata lookup.
# Do NOT copy the full images folder into the container.
RUN mkdir -p /app/data
COPY data/labels.csv ./data/labels.csv

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]