FROM python:3.9
COPY requirements.txt app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt
COPY . /app
EXPOSE 8030
CMD ["streamlit", "run", "app.py","--server.port","8030"]