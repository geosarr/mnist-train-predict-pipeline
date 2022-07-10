# Pulling miniconda3 image
FROM continuumio/miniconda3

# Setting the working directory
WORKDIR /ml_pipeline

# Copying all necessary files to the working directory
COPY train.py .
COPY predict.py .
COPY config.py .
COPY utils.py .

# Creating the python environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activating the environment
SHELL ["conda", "run", "-n", "mnist_env", "/bin/bash", "-c"]

# Training the model and predicting in the environment
CMD python train.py --n_train 40000 --batch_size 128 --n_epochs 10 --n_early_stop 2 --save_losses && python predict.py 
