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

# Copy necessary files/folder to working directory
COPY config.py .
COPY utils.py .
COPY iam.py .
COPY .env .
COPY app /ml_pipeline/app

# Launching the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]