"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='sparse_rev',  # Required
    version='0.1.0',  # Required
    author='Nick Zolman',  # Optional
    author_email='nzolman@uw.edu',  # Optional
    # package_dir={'sparse_rev': 'src/sparse_rev'},  # Optional
    packages= ['sparse_rev'],  # Required
    python_requires='>=3.10, <4',

)