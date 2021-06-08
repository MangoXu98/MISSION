__all__ = ['getTermFeatures', 'termsim', 'funcsim', 'gossfeat', 'proteinfit', 'proteinfct']
__version__ = '15.1'
__release__ = True
__author__ = """Gaston K. Mazandu (gmazandu@{cbio.uct.ac.za, gmail.com}, kuzamunu@aims.ac.za)\n(c) 2015 All rights reserved."""

from dagofun.TermFeatures import getTermFeatures
from dagofun.TermSimilarity import termsim
from dagofun.ProteinSimilarity import funcsim
from dagofun.ProteinSearch import proteinfit
from dagofun.ProteinClustering import proteinfct
from dagofun.EnrichmentAnalysis import gossfeat

