import unittest
import json
from time import sleep
from datetime import timedelta

from fastapi.testclient import TestClient
from jose import jwt
from jose.exceptions import ExpiredSignatureError

from utils import load_dataset
from iam import create_access_token
from app.main import app
from config import SECRET_KEY, ALGORITHM


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.n_train = 40025
        self.batch_size = 64

    def test_load_dataset(self):
        train, val = load_dataset(self.n_train, self.batch_size)
        q, r = self.n_train // self.batch_size, self.n_train % self.batch_size
        assert len(train) == q + (r > 0)
        assert len(val) == 1


class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.gu = {"username": "johndoe", "password": "very_secret"}
        self.bu = {"username": "bozo", "password": "very_bad"}
        self.gu_bsfmts = [self.gu.copy() for _ in range(3)]
        self.gu_bsfmts[0]["scope"] = ["predict"]
        self.gu_bsfmts[1]["scope"] = ["predict:runing"]
        self.gu_bsfmts[2]["scope"] = ["prediction:runing"]
        self.gu_gsfmts_bp = self.gu.copy()
        self.gu_gsfmts_bp["scope"] = ["training:run"]
        self.gu_gsfmts_gp = self.gu.copy()
        self.gu_gsfmts_gp["scope"] = ["prediction:run"]
        self.tok_exp_min = 0.05
        self.endpoint = "prediction"

    def test_post_token(self):
        response_gu = self.client.post("/token", data=self.gu)
        response_bu = self.client.post("/token", data=self.bu)
        responses_gu_bsfmts = [
            self.client.post("/token", data=gu_bsfmt) for gu_bsfmt in self.gu_bsfmts
        ]
        response_gu_gsfmt_bp = self.client.post("/token", data=self.gu_gsfmts_bp)
        response_gu_gsfmt_gp = self.client.post("/token", data=self.gu_gsfmts_gp)

        # good credentials
        assert response_gu.status_code == 200

        # good JWT format
        assert len(json.loads(response_gu.text)["access_token"].split(".")) == 3

        # bad credentials
        assert response_bu.status_code == 401
        assert (
            json.loads(response_bu.text)["detail"] == "Incorrect username or password"
        )

        # bad scope format for good users
        assert responses_gu_bsfmts[0].status_code == 400
        assert responses_gu_bsfmts[1].status_code == 400
        assert responses_gu_bsfmts[2].status_code == 400
        assert response_gu_gsfmt_bp.status_code == 401
        assert response_gu_gsfmt_gp.status_code == 200

        # token expiration
        token = create_access_token(
            data={"sub": self.gu["username"], "scopes": [f"{self.endpoint}:run"]},
            expires_delta=timedelta(minutes=self.tok_exp_min),
        )
        sleep(self.tok_exp_min * 60 + 1.0)
        with self.assertRaises(ExpiredSignatureError):
            jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# import json
# a = TestClient(app)
# r = a.post(
#     "/token",
#     data = {
#         "username": "johndoe",
#         "password": "very_secret",
#         "scope": ["training"]
#     }
# )
# print(json.loads(r.text), r)
