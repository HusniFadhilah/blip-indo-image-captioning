# Step 1: Build Flask App
FROM python:3.11

ARG ENV_FLASK
ENV FLASK_ENV=$ENV_FLASK
ENV FLASK_DEBUG=0
ENV FLASK_APP=app.py
ENV LOG_PATH_APP=./logs/app.log

# Tambahkan dependensi sistem untuk OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Create logs directory
RUN mkdir logs
COPY . /app

COPY requirements.txt .
ENV PIP_NO_CACHE_DIR=1
RUN pip install -r ./requirements.txt


EXPOSE 6600

# Install Gunicorn
# RUN pip install gunicorn
# CMD python ./app.py
# CMD ["gunicorn", "--bind", "0.0.0.0:6600", "app:app"]
CMD ["streamlit", "run", "app.py", "--server.port=6600", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]