import glob
from os.path import basename

import setuptools
from setuptools import find_namespace_packages

setuptools.setup(
    name="anchor", version="0.0.1", author="Megan Leszczynski", packages=find_namespace_packages()
)
