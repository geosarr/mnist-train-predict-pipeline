#!/usr/bin/env python

import sys
import os
from typing import Union
from datetime import timedelta
import sqlite3 as sql

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import  OAuth2PasswordRequestForm
from fastapi.responses import FileResponse

one_level_up = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(one_level_up)
from utils import run_train, load_dataset, run_predict
from config import OAUTH2_SCHEME, ACCESS_TOKEN_EXPIRE_MINUTES
from iam import create_data_user_database, authenticate_user, create_access_token, is_valid_token 


fake_data_db = create_data_user_database()
data_base_path = os.path.join(one_level_up, "data", "users.db")
cursor = sql.connect(data_base_path).cursor()

app = FastAPI()


@app.post("/token")
async def login_with_username_password(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(cursor, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}



@app.get("/training")
def training(
    n_train: int, batch_size: int, n_epochs: int, 
    n_early_stop: int, save_losses: bool, token: str = Depends(OAUTH2_SCHEME)
) -> FileResponse:
    if is_valid_token(token):
        train_loader, val_loader = load_dataset(n_train, batch_size)
        _, __ = run_train(
            n_epochs=n_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            batch_size=batch_size,
            n_train=n_train,
            n_early_stop=n_early_stop,
            save_losses=save_losses,
        )
        return FileResponse(
            path=os.path.join(one_level_up, "results", "losses.png"),
            filename="losses.png",
            media_type="",
        )


@app.get("/prediction")
def prediction(run: bool, token: str = Depends(OAUTH2_SCHEME)
    ) -> Union[FileResponse, None]:
    if run and is_valid_token(token):
        run_predict()
        return FileResponse(
            path=os.path.join(one_level_up, "results", "predictions.txt"),
            filename="predictions.txt",
            media_type="text/plain",
        )


if __name__ == "__main__":
    os.popen("uvicorn main:app --reload --port 8000").read()
