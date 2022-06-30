from setuptools import setup, find_packages

setup(
    name='tyxe',
    version='0.0.1',
    url='https://github.com/karalets/TyXe',
    author=['Hippolyt Ritter', 'Theofanis Karaletsos'],
    author_email='j.ritter@cs.ucl.ac.uk',
    description='BNNs for pytorch using pyro.',
    packages=find_packages(),
    install_requires=[
        'torch >= 1.7.0',
        'torchvision >= 0.8.1',
        'pyro-ppl >= 1.4.0'
    ],
)
