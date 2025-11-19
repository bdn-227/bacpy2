from setuptools import setup, find_packages
import codecs
import os


VERSION = '0.0.25'
DESCRIPTION = 'Bacterial autofluorescence-based classification in python'
LONG_DESCRIPTION = 'Classification of bacteria based on autofluorescence spektra obtained from a tecan plate reader'

# Setting up
setup(
    name="bacpy",
    version=VERSION,
    author="Niklas Kiel (MPIPZ)",
    author_email="<nkiel@mpipz.mpg.de>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    keywords=['python', 'machine learning', 'taxonomic classification', 'bacteria'],
    include_package_data=True,
    package_data={
        "bacpy": ["taxonomy.tsv", "taxonomy2.tsv"],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Biologists",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
