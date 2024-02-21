FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install --upgrade pip &&  \
    pip install gdown && \
    pip install -r requirements.txt

EXPOSE 8032
CMD ["uvicorn", "fast-api:app", "--host", "0.0.0.0", "--port", "8032"]