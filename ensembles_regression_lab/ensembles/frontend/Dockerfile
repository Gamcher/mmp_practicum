FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "ui.py", "--server.port=3000", "--server.address=0.0.0.0"]

