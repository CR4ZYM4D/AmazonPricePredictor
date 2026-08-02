# base image
FROM python:3.13-slim-bookworm

# prevent from buffering logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# copy only requirements first for caching
COPY requirements.txt .

#  dependencies and awscli 
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords

COPY . .


EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]