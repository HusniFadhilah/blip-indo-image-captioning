# Step 1: Build Flask App
FROM python:3.11

ARG ENV_FLASK
ENV FLASK_ENV=$ENV_FLASK
ENV FLASK_DEBUG=0
ENV FLASK_APP=app.py
ENV LOG_PATH_APP=./logs/app.log

WORKDIR /app
# Create logs directory
RUN mkdir logs
COPY . /app

COPY requirements.txt .
RUN pip install -r ./requirements.txt

EXPOSE 6600

# Install Gunicorn
# RUN pip install gunicorn
# CMD python ./app.py
# CMD ["gunicorn", "--bind", "0.0.0.0:6600", "app:app"]
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=6600", "--server.address=0.0.0.0"]