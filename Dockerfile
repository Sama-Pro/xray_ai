FROM python:3.11-slim

WORKDIR /app

COPY . /app

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Render will use PORT env variable
ENV PORT=10000

EXPOSE 10000

CMD ["gunicorn", "deployment.app:app", "--bind", "0.0.0.0:10000"]