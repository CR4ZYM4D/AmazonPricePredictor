from setuptools import find_packages, setup
from typing import List

"""file to setup the basic requirements of the project"""

def get_requirements() -> List[str]:
    requirements = []
    with open('requirements.txt', 'r') as f:

        lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line and not line.endswith('-e .'):
                requirements.append(line)

    return requirements

setup(
    name = "Amazon Item Price Predictor",
    version = "1.0.0",
    author = "Madhav",
    author_email = "devmadhav0207@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)

    