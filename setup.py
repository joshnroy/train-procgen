from setuptools import setup, find_packages

setup(
    name='train-procgen',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'gym==0.15.4',
        'https://github.com/joshnroy/procgen/archive/1.0.zip', # josh's visual procgen
        'tensorflow-gpu==1.15.0',
        'https://github.com/joshnroy/baselines/archive/0.1.zip', # josh's baselines
        'mpi4py==3.0.3',
        'pytest'
    ]
    version='0.0.2',
)
