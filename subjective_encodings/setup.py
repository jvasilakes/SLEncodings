import pathlib
from setuptools import setup, find_packages


here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(name="subjective_encodings",
      version="0.1.0",
      description="A PyTorch extension for encoding and predicting using Subjective Logic.",  # noqa
      long_description=long_description,
      packages=find_packages())
