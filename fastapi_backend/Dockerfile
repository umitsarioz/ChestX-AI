FROM python:3.9

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 8032

CMD ["uvicorn", "fast-api:app", "--host", "0.0.0.0", "--port", "8032"]