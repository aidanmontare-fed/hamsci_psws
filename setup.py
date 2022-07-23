#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='hamsci_grape1',
      version='0.1',
      description='HamSCI Grape v1 Personal Space Weather Station Plotting Tools',
      author='Nathaniel A. Frissell and Kristina V. Collins',
      author_email='nathaniel.frissell@scranton.edu',
      url='https://hamsci.org',
      packages=['hamsci_grape1'],
      install_requires=requirements
     )
