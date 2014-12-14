#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
cracker
~~~~~~~

OpenCL password sha256 hash cracker.

:author: See AUTHORS
"""

from setuptools import setup

setup(
    name='password_cracker',
    version='0.1-dev',
    packages=['cracker'],
    author='Password_Cracker Team',
    author_email='mail@mathias.im',
    url='http://github.com/interru/password_cracker',
    description='A simple opencl sha256 password cracker',
    keywords='Password Hash Cracker',
    long_description=__doc__,
    license='BSD',
    install_requires=[
        'Click',
        'numpy',
        'pyopencl'
    ],
    entry_points='''
        [console_scripts]
        passcracker=cracker:cli
    ''',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ]
)
