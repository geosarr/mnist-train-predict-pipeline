FROM python:3.9.13-slim-buster

# Working directory
WORKDIR /ml_pipeline

# Setting up a virtualenv and its PATH
ENV VIRTUAL_ENV=/ml_pipeline/mnist_env
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install packages from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Exposing a port 
EXPOSE 8080

# Copy necessary files/folder to working directory
COPY train.py .
COPY predict.py .
COPY config.py .
COPY utils.py .
COPY app /ml_pipeline/app

# Launching the API
WORKDIR /ml_pipeline/app
CMD ["uvicorn", "main:app", "--reload", "--port", "8080"]