#!/usr/bin/env python

import sys
import os
from datetime import timedelta
import sqlite3 as sql

from fastapi import Depends, FastAPI, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

one_level_up = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(one_level_up)
from utils import run_train, load_dataset, run_predict
from config import OAUTH2_SCHEME, ACCESS_TOKEN_EXPIRE_MINUTES, USER_PWD_EXCEPTION
from iam import (
    create_data_user_database,
    authenticate_user,
    create_access_token,
    is_valid_token,
)


fake_data_db = create_data_user_database()
data_base_path = os.path.join(one_level_up, "data", "users.db")

app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(one_level_up, "app", "static")),
    name="static",
)


templates = Jinja2Templates(directory=os.path.join(one_level_up, "app", "templates"))


@app.get("/", response_class=HTMLResponse)
def welcome(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/token")
async def login_with_username_password(
    form_data: OAuth2PasswordRequestForm = Depends(),
):  
    cursor = sql.connect(data_base_path).cursor()
    user = authenticate_user(cursor, form_data.username, form_data.password)
    if not user:
        raise USER_PWD_EXCEPTION
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/training")
def training(
    n_train: int,
    batch_size: int,
    n_epochs: int,
    n_early_stop: int,
    save_losses: bool,
    token: str = Depends(OAUTH2_SCHEME),
) -> FileResponse:
    if is_valid_token("training", token):
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
def prediction(token: str = Depends(OAUTH2_SCHEME)) -> FileResponse:
    if is_valid_token("prediction", token):
        run_predict()
        return FileResponse(
            path=os.path.join(one_level_up, "results", "predictions.txt"),
            filename="predictions.txt",
            media_type="text/plain",
        )


if __name__ == "__main__":
    os.popen("uvicorn main:app --reload --port 8000").read()
