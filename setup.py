from setuptools import setup

setup(
    name='online_goban',
    version='0.1',
    description='',
    author="Greg d'Eon",
    author_email='greg.l.deon@gmail.com',
    packages=['online_goban'],  #same as name
    install_requires=['opencv-python'], #external packages as dependencies
)