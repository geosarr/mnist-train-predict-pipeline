import os

from dotenv import load_dotenv
from passlib.context import CryptContext
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Number of classes
N_CLASSES = 10

# Training/Predict input and output data paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Password flow
OAUTH2_SCHEME = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={"training": "Run", "predicting": "Run"}
)

# Password management
PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")

# API Token generation parameters
load_dotenv()
SECRET_KEY = os.environ["JWT_SECRET_KEY"]
ALGORITHM = os.environ["JWT_SIG_ALGORITHM"]
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ["JWT_DURATION_MINUTES"])

# Exceptions
CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

EXPIRATION_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Expired token",
    headers={"WWW-Authenticate": "Bearer"},
)
