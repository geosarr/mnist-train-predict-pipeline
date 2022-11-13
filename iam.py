import os
from typing import Union
from pydantic import BaseModel
from datetime import datetime, timedelta

from fastapi import Depends
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
import sqlite3 as sql
from config import DATA_DIR, PWD_CONTEXT, SECRET_KEY, ALGORITHM, OAUTH2_SCHEME
from config import  CREDENTIALS_EXCEPTION, EXPIRATION_EXCEPTION, SCOPE_EXCEPTION


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str
    scopes: dict


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return PWD_CONTEXT.verify(plain_password, hashed_password)


def get_user(cursor: sql.Cursor, username: str) -> Union[UserInDB, None]:
    db_user = cursor.execute(
        f"""
        SELECT username, hashed_password, training , prediction 
        FROM users
        WHERE username = '{username}'
    """
    ).fetchall()
    if len(db_user) == 1:
        db_user = {
            "username": db_user[0][0], 
            "hashed_password": db_user[0][1],
            "scopes": {
                "training": db_user[0][2],
                "prediction": db_user[0][3]
            }        
        }
        return UserInDB(**db_user)


def authenticate_user(cursor, username: str, password: str) -> Union[bool, UserInDB]:
    user = get_user(cursor, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def is_valid_token(endpoint: str, token: str = Depends(OAUTH2_SCHEME)) -> bool:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        scopes: dict = payload.get("scopes")
        if username is None:
            raise CREDENTIALS_EXCEPTION
        if scopes[endpoint] != "run":
            raise SCOPE_EXCEPTION
    except ExpiredSignatureError as exc1:
        raise EXPIRATION_EXCEPTION from exc1
    except JWTError as exc2:
        raise CREDENTIALS_EXCEPTION from exc2

    return True


def create_data_user_database() -> None:
    """
    Creates a database of users of the API
    Args:
        None
    Returns:
        None
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    users_database_path = os.path.join(DATA_DIR, "users.db")
    if not os.path.isfile(users_database_path):
        con = sql.connect(users_database_path)
        cur = con.cursor()

        cur.execute(
            """CREATE TABLE users 
                    (username text, hashed_password text, training text, prediction text)"""
        )

        cur.execute(
            """INSERT INTO users VALUES
                    ("johndoe","$2b$12$4FdZ1IT6QarM5CyQ7DEx.e/pwLQQGBpVQLgWVyMs3DnkAHTCOys6W", ' ', "run"),
                    ("alice", "$2b$12$UqWFgNQXSGDFsQs6fnHjTeCcPfa1eSGGVrBSpgk4/X/s/aaN/O6Hy", "run", "run")"""
        )

        con.commit()
        con.close()
