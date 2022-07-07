In this repository, a relatively simple MLP is implemented to predict the MNIST test set from http://yann.lecun.com/exdb/mnist/ after training it on a subset of the training set and validating it on the remaining subset of the training set.

# Installation
## Using conda
-- Install miniconda3

-- Reproduce the python environment using `conda env create -f environment.yml` or lauching the following commands :
    `conda create -n mnist_env python=3.9` then `pip install -r requirements.txt`

-- Activate the environment using `conda activate mnist_env`

-- Launch `python train.py --help` to see how to train the model

-- Launch `python predict.py` to do the prediction


## Using the Dockerfile
-- Install the `docker` command

-- launch `docker build -t [name_image]` then `docker run [name_image]` (`[name_image]` being a name of your choice (as long as it is acceptable) that has to be used for both command lines.)