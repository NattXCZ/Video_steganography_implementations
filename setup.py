from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Video steganography implementation of three methods',
    version='0.1',
    author='Natalie Trhlikova',
    author_email='natalie.trhlikova01@upol.cz',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'opencv-python',
        'pywavelets',
        'galois',
        'scikit-image',
    ],
    python_requires='>=3.9',
)