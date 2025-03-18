from setuptools import setup, find_packages

setup(
    name='runner_al',
    description='A library for running active learning with RuNNer 4G models',
    author='Md Omar Faruque',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'mpi4py',
        'numpy',
    ],
)
