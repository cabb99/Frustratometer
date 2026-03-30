"""
DCA Frustratometer
Calculates single residue frustration, and mutational frustration of proteins.
"""
import sys
from setuptools import setup, find_packages
import versioneer

short_description = "Calculates single residue frustration, and mutational frustration of proteins.".split("\n")[0]

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = None


setup(
    # Self-descriptive entries which should always be present
    name='frustratometer',
    author='Carlos Bueno',
    author_email='carlos.bueno@rice.edu',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    python_requires=">=3.7",
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'biopython',
        'prody',
        'pyparsing',
        'pydantic>=2',
        'numba',
    ],

)
