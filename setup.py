from setuptools import setup, find_packages

setup(
    name='AliO',
    version='0.1.0',
    description='Official implementation of AliO',
    packages=find_packages(),
    author='Anonymous',
    install_requires=[
        'torch',
    ],
)