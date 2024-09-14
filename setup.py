from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e.'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path,'r') as f:
        requirements = f.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

setup(
    name='stroke-prediction-ml-project',
    version='0.1',
    author='Ayush Bhattacharya',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)