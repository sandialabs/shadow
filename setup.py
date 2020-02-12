from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='shadow-ssml',
    version='1.0.0',
    author="Shadow Developers",
    author_email="dzander@sandia.gov",
    maintainer_email="dzander@sandia.gov",
    description="Semi-supervised machine learning for PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandialabs/shadow",
    license='Revised BSD',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'torch>1',
        'numpy'
    ],
    python_requires=">=3.6"
)
