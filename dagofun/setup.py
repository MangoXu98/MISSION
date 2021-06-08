#!/usr/bin/env python

from distutils.core import setup
from platform import python_version_tuple
import re

LICENSE = open("LICENSE").read()

# strip links from the descripton on the PyPI
LONG_DESCRIPTION = open("README.rst").read().replace("`_", "`")

# strip Build Status from the PyPI package
if python_version_tuple()[:2] >= ('2', '7'):
    LONG_DESCRIPTION = re.sub("^Build status\n(.*\n){7}", "", LONG_DESCRIPTION, flags=re.M)



setup(name='A-DaGO-Fun',
   version='15.1',
   description='Computing GO-based semantic similarity scores and includes several \nother biological applications related to GO semantic similarity measures',
   long_description=LONG_DESCRIPTION,
   author='Gaston K. Mazandu et al.',
   author_email='gmazandu@cbio.uct.ac.za, emile@cbio.uct.ac.za, mamana@aims.ac.za, nicola.mulder@uct.ac.za',
   maintainer = 'Gaston K. Mazandu',
   maintainer_email = 'gmazandu@{cbio.uct.ac.za, gmail.com}, kuzamunu@aims.ac.za',
   url='http://web.cbio.uct.ac.za/ITGOM/adagafun',
   license=LICENSE,
   classifiers= [ "Development Status :: 4 - Beta",
                  "License :: OSI Approved :: GNU General Public License",
                  "Operating System :: OS Independent, but tested only on Linux",
                  "Programming Language :: Python :: Not tested on the version less than 2.7.3",
                  "Programming Language :: Python :: 2.7.3",
                  "Topic :: Software Development :: Libraries",
                  "Following libraries need to be installed prior to the installation and the\nuse of A-DaFO-Fun:",
                   "\t::scipy\n\t::cPickle\n\t::networkx\n\t::matplotlib"],
   platforms = 'Linux',
   requires = ['\nscipy', 'networkx', 'matplotlib\n'],
   py_modules = ['TermFeatures', 'TermSimilarity', 'ProteinSimilarity', 'ProteinSearch', 'EnrichmentAnalysis', 'ProteinClustering'],
   package_data={'dagofun': ['data/*.ck'],})
