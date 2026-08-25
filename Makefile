.PHONY: build install test clean

build:
	python3 -m pip install -e .

install:
	python3 -m pip install -e ".[dev]"

test:
	PYTHONPATH=. pytest -v

clean:
	rm -rf build/ dist/ *.egg-info boptrace/*.so boptrace/*.pyd cpp/*.o cpp/*.obj
	find . -type d -name "__pycache__" -exec rm -rf {} +
