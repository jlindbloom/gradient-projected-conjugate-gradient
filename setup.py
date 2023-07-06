from setuptools import setup, find_packages

setup(
    name='gpcg',
    version='0.0.1',
    author='Jonathan Lindbloom',
    author_email='jonathan@lindbloom.com',
    license='LICENSE',
    packages=find_packages(),
    description='A simple solver for bound-constrained quadratic programs using a gradient-projected conjugate gradient algorithm.',
    long_description=open('README.md').read(),
)