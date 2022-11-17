#!/usr/bin/env python

import sys
import os
from datetime import timedelta
import sqlite3 as sql

from fastapi import Depends, FastAPI, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

one_level_up = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(one_level_up)
from utils import run_train, load_dataset, run_predict
from config import (
    OAUTH2_SCHEME,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    USER_PWD_EXCEPTION,
    ENDPOINTS,
)
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
    if len(form_data.scopes) > 0:
        # user inputs scopes: check first format of scopes and finally check permissions
        final_scopes = []
        for scope in form_data.scopes:
            scope_info = scope.strip().split(":")
            if len(scope_info) != 2:
                # format should be for e.g training:run
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Bad scope {scope}, should be of the form endpoint_name:permission, e.g. training:run",
                )
            elif scope_info[0] not in ENDPOINTS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Bad scope {scope}, the only available endpoints are {OAUTH2_SCHEME.scopes}, instead of {scope_info[0]}",
                )
            elif scope_info[1] != "run":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Bad scope {scope}, the only available permission is run instead of {scope_info[1]}",
                )
            else:
                # check whether or not the user has this permission
                if scope.strip() not in user.scopes:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"No scope {scope} for user",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            final_scopes.append(scope)
    else:
        # If the user does not specify scopes, the hard-coded ones in the user database will be used
        final_scopes = user.scopes
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": final_scopes},
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
