from setuptools import setup, find_packages

setup(
    name='train-procgen',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'gym==0.15.4',
        'tensorflow-gpu==1.15.0',
        'mpi4py==3.0.3',
        'pytest',
        'procgen',
        'baselines',
    ],
    dependency_links = [
        'git@github.com:joshnroy/procgen.git@1.0#egg=procgen', # visual procgen
        'git@github.com:joshnroy/baselines.git@0.1#egg=baselines', # baselines
    ],
    version='0.0.2',
)
