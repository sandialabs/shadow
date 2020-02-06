from setuptools import setup, find_packages

setup(
    name='shadow-ssml',
    version='1.0.0',
    license='BSD 3-Clause',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'torch>1',
        'numpy'
    ]
)
