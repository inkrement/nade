#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from io import open
from os import path

from setuptools import find_packages, setup

with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'),
          encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nade',
    version='0.0.1',
    author='Christian Hotz-Behofsits',
    author_email='chris.hotz.behofsits@gmail.com',
    url='https://github.com/inkrement/nade',
    description='NADE (natural affect detection) allows to infer basic emotions from textual messages',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='affect detection, emotions',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,
    python_requires='>= 3.6',
    install_requires=["fasttext", "numpy"],
    license='MIT'
)