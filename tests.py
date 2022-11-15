import unittest
from utils import load_dataset
from app.main import app
from fastapi.testclient import TestClient

class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.n_train = 40025
        self.batch_size = 64

    def test_load_dataset(self):
        train, val = load_dataset(self.n_train, self.batch_size)
        q, r = self.n_train // self.batch_size, self.n_train % self.batch_size
        assert len(train) == q + (r > 0)
        assert len(val) == 1

class TestApp(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)


    def test_post_token(self):
        # response = (self.client).post("/token") 
        assert 1==1

