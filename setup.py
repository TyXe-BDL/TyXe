from setuptools import setup, find_packages

setup(
    name='tyxe',
    version='0.0.2',
    url='https://github.com/TyXe-BDL/TyXe',
    author=['Hippolyt Ritter', 'Theofanis Karaletsos'],
    author_email='j.ritter@cs.ucl.ac.uk',
    description='BNNs for pytorch using pyro.',
    packages=find_packages(),
    install_requires=[
        'torch == 1.12.0',
        'torchvision == 0.13.0',
        'pyro-ppl == 1.8.1'
    ],
)
