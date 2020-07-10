from setuptools import setup, find_packages

setup(
    name='tyxe',
    version='0.0.1',
    url='https://github.com/ucl-jr/TyXe',
    author=['Hippolyt Ritter', 'Theofanis Karaletsos'],
    author_email='j.ritter@cs.ucl.ac.uk',
    description='Description of my package',
    packages=find_packages(),
    install_requires=['pyro-ppl >= 1.3.0'],
)
