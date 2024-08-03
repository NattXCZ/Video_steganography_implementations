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
        'numpy==2.0.0',
        'opencv-python==4.10.0.84',
        'pywavelets==1.6.0',
        'galois==0.4.1',
        'scikit-image==0.24.0',
        'scipy==1.13.1',
        'pillow==10.4.0',
        'networkx==3.2.1',
        'numba==0.60.0',
        'llvmlite==0.43.0',
    ],
    python_requires='>=3.9',
)