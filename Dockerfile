FROM python:3.10-slim
WORKDIR /app
# copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn numpy pyyaml openai

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
