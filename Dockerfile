FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir ".[server]"

EXPOSE 8000

CMD ["python", "-m", "ghostfolio_agent.server"]
