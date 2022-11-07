install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint: 
	pylint --disable=R,C,E1101,W1309 predict train utils
	# pylint predict train utils

test: 
	pytest tests.py 

all: install format lint test

		

