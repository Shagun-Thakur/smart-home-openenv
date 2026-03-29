FROM python:3.10-slim
WORKDIR /app
# copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir numpy pyyaml

CMD ["python", "-m", "scripts.evaluate"]