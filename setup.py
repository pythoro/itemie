# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:02:12 2019

@author: Reuben
"""

import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="itemie",
    version="0.0.1",
    author="Reuben Rusk",
    author_email="pythoro@mindquip.com",
    description="Object-oriented analysis for surveys and questionnaires.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pythoro/itemie.git',
    project_urls={
        'Documentation': 'https://itemie.readthedocs.io/en/latest/',
        'Source': 'https://github.com/pythoro/itemie.git',
        'Tracker': 'https://github.com/pythoro/itemie/issues',
    },
    download_url="https://github.com/pythoro/itemie/archive/v0.1.1.zip",
    packages=['itemie'],
    keywords=['NUMPY', 'SURVEY', 'QUESTIONNAIRE'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=['numpy', 'pandas'],
)