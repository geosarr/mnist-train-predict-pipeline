import unittest
import json
from utils import load_dataset
from app.main import app
from fastapi.testclient import TestClient


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
        self.data_john = {"username": "johndoe", "password": "very_secret"}
        self.data_fake = {"username": "bozo", "password": "very_bad"}

    def test_post_token(self):
        response0 = self.client.post("/token", data=self.data_john)
        response1 = self.client.post("/token", data=self.data_fake)
        # good credentials
        assert response0.status_code == 200
        assert (
            len(json.loads(response0.text)["access_token"].split(".")) == 3
        )  # JWT format
        # bad credentials
        assert response1.status_code == 401
        assert json.loads(response1.text)["detail"] == "Incorrect username or password"


# import json
# a = TestClient(app)
# r = a.post(
#     "/token",
#     data = {
#         "username": "johndoe",
#         "password": "very_secret"
#     }
# )
# print(json.loads(r.text))
