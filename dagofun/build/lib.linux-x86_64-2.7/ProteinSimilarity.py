#!/usr/bin/env python

"""
This python file is part of the DaGO-Fun tool, which is a tool for Gene 
Ontology-based functional analysis using term information content 
measures.
This particular python code implements all known Gene Ontology IC-based 
protein functional similarity measures and allows to use more than one 
approach. In fact, up to three measures can be simultaneously run. This
can run over any set of annotated proteins, not only those that are
found in the UniProtKB-GOA dataset.

The main website for the A-DaGO-Fun package is 
http://web.cbio.uct.ac.za/ITGOM/adagofun where users can find essential 
information about obtaining G-DaGO-Fun. It is freely downloadable under 
GNU General Public License (GPL), pre-compiled for Linux version and pro-
tected by copyright laws. Users are free to copy, modify, merge, publish,
distribute and display information contained in the package, provided 
that it is done with appropriate citation of the package and by
including the permission notice in all copies or substantial portions of 
the module contained in this package.

DaGO-Fun is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED 
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS 
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY WHETHER IN AN 
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
See <http://www.gnu.org/licenses/>.

This code was written by 
    Gaston K. Mazandu <gmazandu@{cbio.uct.ac.za, gmail.com}, 
                       kuzamunu@aims.ac.za>
    (c) 2015 under free software (GPL) All rights reserved.
"""

__all__ = ["funcsim"]
__version__= "15.1"
__author__ = """Gaston K. Mazandu <gmazandu@{cbio.uct.ac.za, gmail.com}, kuzamunu@aims.ac.za>"""

# Importing necessary python libraries
import sys, os, re, inspect, time

# Importing external python modules and libraries
from tabulate import tabulate as tab
from functools import reduce
try: # It is sufficient to import "cPickle" here once for reading or storing python binary files
	import pickle 
except ImportError:
    import pickle as cPickle

try:
	from numpy import array, exp, mean, sqrt, dot
except ImportError:
    raise ImportError("The library SciPy is required for reading binary files. \nPlease, install it and try again ...")

# Defining necessary global variables
goic, lcc, gocf, icc, termcc = {}, {}, {}, {}, {}
Head = None; MaxValue = None
TermSim = {}; protgo = {}
Fam = ['AnnChar', 'Universal', 'WangIC', 'Zhang']
Ont = ['BP', 'MF', 'CC']

# Function definitions start here!
def getGOdata(so):
	"""
		For loading all gene ontology features for the ontology under consideration.
	"""
	global Head, lcc, Ont, icc, termcc
	lcc =  pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sLevelAncestor.ck'%(Ont[so-1], )),'rb'))
	icc = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sTermIndex.ck'%(Ont[so-1], )),'rb'))
	termcc = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sTerms.ck'%(Ont[so-1], )),'rb'))
		
	if so==3: Head = icc['GO:0005575']
	elif so==2: Head = icc['GO:0003674']
	else: Head = icc['GO:0008150']	
	return

def getICdata(so, sf, drop = 0):
	"""
		For loading all Gene Ontology term  Information Content values for a given ontology and approach.
	"""
	global goic, gocf, Ont, Fam
	goic.clear(); gocf.clear()
		
	if sf==1 and drop==1:
		goic = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%s%sPartialI.ck'%(Ont[so-1], Fam[sf-1])),'rb'))
	else:
		goic = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%s%sI.ck'%(Ont[so-1], Fam[sf-1])),'rb'))
	
	if sf==3: 
		gocf = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sWangCFI.ck'%(Ont[so-1],)),'rb'))
	return

def retrieveTermSimilarity(p, app):
	global Head, lcc, goic, gocf, TermSim, protgo, MaxValue
	if app==1: # For GO-Universal
		TermSim.clear()
		tt = [(a, b) for a in protgo[p[0]] for b in protgo[p[1]]]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				mica = max([goic[t] for t in ancestt])
				TermSim[(a,b)] = TermSim[(b,a)] = mica/max(goic[a],goic[b])
				del ancestt, mica
	elif app==2: # For Wang et al.
		TermSim.clear()
		tt = [(a, b) for a in protgo[p[0]] for b in protgo[p[1]]]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(gocf[a].keys()).intersection(list(gocf[b].keys()))
				TermSim[(a, b)] = TermSim[(b, a)] = sum([gocf[a][t]+gocf[b][t] for t in ancestt])/(goic[a]+goic[b])
				del ancestt
		del tt
	elif app==3: # For Zhang et al
		TermSim.clear()
		tt = [(a, b) for a in protgo[p[0]] for b in protgo[p[1]]]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				mica = max([goic[t] for t in ancestt])
				TermSim[(a, b)] = TermSim[(b, a)] = 2*mica/(goic[a]+goic[b])
				del ancestt, mica
	elif app == 4: # Resnik-XGraSM
		TermSim.clear()
		MaxValue = max(goic.values())
		ssource = protgo[p[0]] & set(goic.keys()); ddest = protgo[p[1]] & set(goic.keys())
		tt = [(a, b) for a in ssource for b in ddest]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				ancestt = ancestt.difference([Head])
				mica = mean([goic[t] for t in ancestt]) if ancestt else 0.0
				TermSim[(a,b)] = TermSim[(b,a)] = mica/MaxValue
				del ancestt, mica
	elif app == 5: # Nunivers-XGraSM
		TermSim.clear()
		ssource = protgo[p[0]] & set(goic.keys()); ddest = protgo[p[1]] & set(goic.keys())
		tt = [(a, b) for a in ssource for b in ddest]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				ancestt = ancestt.difference([Head])
				mica = mean([goic[t] for t in ancestt]) if ancestt else 0.0
				TermSim[(a,b)] = TermSim[(b,a)] = mica/max(goic[a], goic[b])
				del ancestt, mica
	elif app == 6: # Lin-XGraSM
		TermSim.clear()
		ssource = protgo[p[0]] & set(goic.keys()); ddest = protgo[p[1]] & set(goic.keys())
		tt = [(a, b) for a in ssource for b in ddest]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				ancestt = ancestt.difference([Head])
				mica = mean([goic[t] for t in ancestt]) if ancestt else 0.0
				TermSim[(a,b)] = TermSim[(b,a)] = 2*mica/(goic[a]+goic[b])
				del ancestt, mica
	elif app == 7: # Resnik
		TermSim.clear()
		ssource = protgo[p[0]] & set(goic.keys()); ddest = protgo[p[1]] & set(goic.keys())
		tt = [(a, b) for a in ssource for b in ddest]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				mica = max([goic[t] for t in ancestt])
				TermSim[(a, b)]= TermSim[(b, a)]  = mica/MaxValue
				del ancestt, mica
	elif app == 8: # Nunivers
		TermSim.clear()
		ssource = protgo[p[0]] & set(goic.keys()); ddest = protgo[p[1]] & set(goic.keys())
		tt = [(a, b) for a in ssource for b in ddest]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				mica = max([goic[t] for t in ancestt])
				TermSim[(a,b)] = TermSim[(b,a)] = mica/max(goic[a], goic[b])
				del ancestt, mica
	elif app == 9: # Lin
		TermSim.clear()
		ssource = protgo[p[0]] & set(goic.keys()); ddest = protgo[p[1]] & set(goic.keys())
		tt = [(a, b) for a in ssource for b in ddest]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				mica = max([goic[t] for t in ancestt])
				TermSim[(a,b)] = TermSim[(b,a)] = 2*mica/(goic[a]+goic[b])
				del ancestt, mica
	elif app == 10: # SimRel
		TermSim.clear()
		ssource = protgo[p[0]] & set(goic.keys()); ddest = protgo[p[1]] & set(goic.keys())
		tt = [(a, b) for a in ssource for b in ddest]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				mica = max([goic[t] for t in ancestt])
				TermSim[(a, b)] = TermSim[(b,a)] = 2*mica*(1.0-exp(-mica))/(goic[a]+goic[b])
				del ancestt, mica
	elif app == 11: # Li et al.
		TermSim.clear()
		ssource = protgo[p[0]] & set(goic.keys()); ddest = protgo[p[1]] & set(goic.keys())
		tt = [(a, b) for a in ssource for b in ddest]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				mica = max([goic[t] for t in ancestt])
				TermSim[(a, b)] = TermSim[(b, a)] = 2*mica*(1.0-1.0/(1.0+mica))/(goic[a]+goic[b])
				del ancestt, mica
	return

def computingProteinSimilarityScore(p, sem):
	global TermSim, protgo, Head, lcc, goic
	data = None
	if sem==1: # SimGIC
		try:
			p0terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[0]]])
			p1terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[1]]])
			p0terms = p0terms.difference([Head]) & set(goic.keys())
			p1terms = p1terms.difference([Head]) & set(goic.keys())
			sanc = sum([goic[t] for t in p0terms.intersection(p1terms)])/sum([goic[t] for t in p0terms.union(p1terms)])
			data = round(sanc,5)	
			del p0terms, p1terms, sanc
		except:
			pass
	elif sem==2: # SimDIC
		try:
			p0terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[0]]])
			p1terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[1]]])
			p0terms = p0terms.difference([Head]) & set(goic.keys())
			p1terms = p1terms.difference([Head]) & set(goic.keys())
			sanc1 = sum([goic[t] for t in p0terms.intersection(p1terms)]); sanc2 = sum([goic[t] for t in p0terms.union(p1terms)])
			data = round(2.0*sanc1/(sanc2 + sanc1),5)
			del p0terms, p1terms, sanc1, sanc2
		except:
			pass
	elif sem==3: # SimUIC
		try:
			p0terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[0]]])
			p1terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[1]]])
			p0terms = p0terms.difference([Head]) & set(goic.keys())
			p1terms = p1terms.difference([Head]) & set(goic.keys())
			sanc1 = sum([goic[t] for t in p0terms.intersection(p1terms)]); sanc2 = sum([goic[t] for t in p0terms])
			sanc3 = sum([goic[t] for t in p1terms])
			data = round(sanc1/max(sanc2, sanc3),5)
			del p0terms, p1terms, sanc1, sanc2, sanc3
		except:
			pass
	elif sem==4: #SimCOU
		try:
			p0terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[0]]])
			p1terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[1]]])
			p0terms = p0terms.difference([Head]); p1terms = p1terms.difference([Head]);
			tunion = list((p0terms | p1terms) & set(goic.keys()))
			sanc1 = array([goic[t] if t in p0terms else 0.0 for t in tunion])
			sanc2 = array([goic[t] if t in p1terms else 0.0 for t in tunion])
			data = dot(sanc1, sanc2)/(sqrt(dot(sanc1, sanc1))*sqrt(dot(sanc2, sanc2)))
			del p0terms, p1terms, sanc1, sanc2, tunion
		except:
			pass
	elif sem==5: #SimCOT
		try:
			p0terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[0]]])
			p1terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[1]]])
			p0terms = p0terms.difference([Head]); p1terms = p1terms.difference([Head]);
			tunion = list((p0terms | p1terms) & set(goic.keys()))
			sanc1 = array([goic[t] if t in p0terms else 0.0 for t in tunion])
			sanc2 = array([goic[t] if t in p1terms else 0.0 for t in tunion])
			data = dot(sanc1, sanc2)/(dot(sanc1, sanc1)+dot(sanc2, sanc2)-dot(sanc1, sanc2))
			del p0terms, p1terms, sanc1, sanc2, tunion
		except:
			pass
	elif sem==6: # SimUI
		try:
			p0terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[0]]])
			p1terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[1]]])
			p0terms = p0terms.difference([Head]); p1terms = p1terms.difference([Head]);
			sanc = 1.0*len(p0terms.intersection(p1terms))/len(p0terms.union(p1terms))
			data = round(sanc,5)
			del p0terms, p1terms, sanc	
		except:
			pass
	elif sem==7: # SimUB
		try:
			p0terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[0]]])
			p1terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[1]]])
			p0terms = p0terms.difference([Head]); p1terms = p1terms.difference([Head]);
			sanc = 1.0*len(p0terms.intersection(p1terms))/max(len(p0terms), len(p1terms))
			data = round(sanc,5)
			del p0terms, p1terms, sanc	
		except:
			pass
	elif sem==8: # SimDB
		try:
			p0terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[0]]])
			p1terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[1]]])
			p0terms = p0terms.difference([Head]); p1terms = p1terms.difference([Head]);
			sanc = 2.0*len(p0terms.intersection(p1terms))/(len(p0terms)+len(p1terms))
			data = round(sanc,5)
			del p0terms, p1terms, sanc	
		except:
			pass
	elif sem==9: # SimNTO
		try:
			p0terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[0]]])
			p1terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[1]]])
			p0terms = p0terms.difference([Head]); p1terms = p1terms.difference([Head]);
			sanc = 1.0*len(p0terms.intersection(p1terms))/min(len(p0terms), len(p1terms))
			data = round(sanc,5)
			del p0terms, p1terms, sanc	
		except:
			pass
	elif sem==10: # Average
		try:
			fset = protgo[p[0]] & set(goic.keys()); sset = protgo[p[1]] & set(goic.keys())
			au = [TermSim[(s,t)] for t in fset for s in sset]
			data = round(sum(au)/len(au),5)
			del au
		except:
			pass
	elif sem==11: # Best Match Average
		try:
			fset = protgo[p[0]] & set(goic.keys()); sset = protgo[p[1]] & set(goic.keys())
			au1 = [max([TermSim[(s,t)] for t in sset]) for s in fset]
			au2 = [max([TermSim[(s,t)] for t in fset]) for s in sset]
			data = round((mean(au1)+mean(au2))/2.0,5)
			del au1, au2
		except:
			pass
	elif sem==12: # Average Best Matches
		try:
			fset = protgo[p[0]] & set(goic.keys()); sset = protgo[p[1]] & set(goic.keys())
			au1 = [max([TermSim[(s,t)] for t in sset]) for s in fset]
			au2 = [max([TermSim[(s,t)] for t in fset]) for s in sset]
			au = au1+au2
			data = round(mean(au),5)
			del au1, au2, au
		except:
			pass
	elif sem==13: # Best Match Maximum (RCMax) or MHDF
		try:
			fset = protgo[p[0]] & set(goic.keys()); sset = protgo[p[1]] & set(goic.keys())
			au1 = [max([TermSim[(s,t)] for t in sset]) for s in fset]
			au2 = [max([TermSim[(s,t)] for t in fset]) for s in sset]
			data = round(max(mean(au1), mean(au2)),5)
			del au1, au2
		except:
			pass
	elif sem==14: # HDF
		try:
			fset = protgo[p[0]] & set(goic.keys()); sset = protgo[p[1]] & set(goic.keys())
			au1 = [max([TermSim[(s,t)] for t in sset]) for s in fset]
			au2 = [max([TermSim[(s,t)] for t in fset]) for s in sset]
			data = round(1.0-max(1.0-min(au1), 1.0-min(au2)),5)
			del au1, au2
		except:
			pass
	elif sem==15: # VHDF
		try:
			fset = protgo[p[0]] & set(goic.keys()); sset = protgo[p[1]] & set(goic.keys())
			au1 = [(1.0-max([TermSim[(s,t)] for t in sset]))**2 for s in fset]
			au2 = [(1.0-max([TermSim[(s,t)] for t in fset]))**2 for s in sset]
			data = round(1.0-(sqrt(sum(au1)/len(au1)) + sqrt(sum(au1)/len(au1)))/2.0, 5)
			del au1, au2
		except:
			pass
	elif sem==16: # Maximum
		try:
			fset = protgo[p[0]] & set(goic.keys()); sset = protgo[p[1]] & set(goic.keys())
			au = [TermSim[(s,t)] for t in fset for s in sset]
			data = round(max(au),5)
			del au
		except:
			pass
	return data

def readProteinFile(FileName):
	""" 
		Reading protein file and constructing protgo dictionary from the database, it takes 3 parameters and return 2 values:
		Inputs: FileName is the name of file containing protein pairs, Id is the protein Identifiers 1 for UniProt Ids and 2 for Genenames
				  and so indicating the ontology undeer consideration!
		Outputs: 
	"""
	global protgo, icc, termcc, Head
	
	try:
		fp = open(os.path.join(os.path.dirname(__file__),FileName),'r')
	except:
		print("Check the path to your file or if the file exists.\nFor now, the process cannot be pursued: reading file problem ...\n")
		sys.exit(0)

	for line in fp:
		ligne = line.strip()
		if not ligne: continue
		ligne = ligne.split()
		if len(ligne)!=2: continue
		protein = ligne[0].strip()
		termprotein = set([x.strip() for x in ligne[1].split(',')])
		protgo[protein] = set([icc[t] for t in termprotein if (t in icc) and termcc[icc[t]][-1] and icc[t]!=Head])
		del termprotein
	fp.close()

def inputpairs(FileName):
	# Reading protein or gene pairs from a file
	try:
		pairdata = []
		fp = open(os.path.join(os.path.dirname(__file__),FileName), 'r')
		for line in fp:
			ligne = line.strip()
			if not ligne or ligne.startswith('#'): continue
			ligne = ligne.split()
			if len(ligne)!=2: continue
			pairdata.append(tuple([x.strip() for x in ligne]))
		fp.close()
		return pairdata
	except:
		print("Check the path to your file or if the file exists.\nFor now, the process cannot be pursued: reading file problem ...\n")
		sys.exit(0)

def _fixkwargs(dicts, allapp):
	"""
		This module checks different parameters *args and **kwargs and align them correctly in order to run funcsim module.
	"""
	
	if len(dicts) > 5: 
		print("5 or 6 arguments are required but more than 6 arguments provided.\nFor now, the process cannot be pursued: number of arguments error ...") #raise IOError
		sys.exit(1)
	
	kwargs = {}
	if 'Targets' not in dicts: kwargs['Targets'] = []
	elif isinstance(dicts['Targets'], (str, list, tuple)): kwargs['Targets'] = dicts['Targets']
	else:
		print("Check list of protein/gene pairs.\nFor now, the process cannot be pursued: error on input data...")
		sys.exit(0)
	if 'ontology' not in dicts: kwargs['ontology'] = 'BP'
	elif dicts['ontology'].upper() in ['BP','MF','CC']: kwargs['ontology'] = dicts['ontology'].upper()
	else:
		print("Check your ontology, the ProteinSimilarity module uses the string:\n\t'BP': For Biological Process\n\t'MF': For Molecular Function\n\t'CC': For Cellular Component.\nunknown ontology was provided")
		sys.exit(1)

	if 'measure' not in dicts: kwargs['measure'] = ('ubma',)
	elif type(dicts['measure'])==str: 
		if dicts['measure'].lower() in allapp: kwargs['measure'] = (dicts['measure'].lower(),)
		else:
			print("Check notations of different measures.\nFor now, the process cannot be pursued: measure key error ...")
			sys.exit(0)
	elif isinstance(dicts['measure'], (tuple, list)):
		dicts['measure'] = [p.lower() for p in dicts['measure']]
		if set(dicts['measure']).issubset(allapp):
			kwargs['measure'] = (dicts['measure'][0],)
			for i in range(1,len(dicts['measure'])):
				if not dicts['measure'][i] in kwargs['measure'] and len(kwargs['measure']) < 4: 
					kwargs['measure'] += (dicts['measure'][i],)
		else:
			print("Check notations of different measures.\nFor now, the process cannot be pursued: measure key error ...")
			sys.exit(1)
	else:
		print("Measure key is either a symbol of measure, list or tuple of measure symbols.\nCheck measure symbols and try again ...")
		sys.exit(1)

	if 'drop' not in dicts: kwargs['drop'] = 0
	elif dicts['drop'] is 0 or dicts['drop'] is 1: kwargs['drop'] = dicts['drop']
	else:
		print("Check the use of IEA evidence code variable <drop> which should be a Boolean:\n\t0 if all evidence code should be used and \n\t1 if IEA evidence code should be excluded.\n\nPlease check and try again ...") # raise ValueError
		sys.exit(1)

	if 'output' not in dicts: kwargs['output'] = 1
	elif dicts['output'] is 0 or dicts['output'] is 1 or dicts['output'] is 2: kwargs['output'] = dicts['output']
	else:
		print("How do you want to output results is an Enum 0, 1, 2:\n\t1 if results should be displayed on the screen \n\t0 if results should be written in a file.\n\t2 for outputting a Python object for possible further usage.\n\nPlease check and try again ...") # raise ValueError
		sys.exit(1)
	kwargs['tablefmt'] = 'rst' # One can also use 'grid' or other display formats!
	return kwargs

	
def funcsim(*args, **kwargs):
	"""
This function retrieves protein functional (semantic) similarity scores 
between proteins.
*args* is a variable length argument, which is mainly a string 
representing the name of the file containing protein IDs and their 
associated GO IDs and possibly the list of proteins/genes for which 
functional similarity (SS) scores must be retrieved. 

For example, the following is legal::

>>> funcsim('path/to/file_of_ProteinIDs_GOIDs') # displays the SS value
 beween protein pairs of annotated proteins found in the file using a 
default ontology: biological process, a default SS approach: 
GO-universal based best match average measure, Considering all GO 
evidence codes (all = 1) and display by default on 
the screen (output=1), and finally default table display 
(tablefmt="rst") see tabulate package written by 'Sergey Astanin 
(s.astanin@gmail.com)' and collaborators.

*kwargs* can be used to set ontology, measure, drop, outputs and 
tablefmt arguments. Here are some examples::
    	 
>>> funcsim('path/to/file_ProtIDs_GOIDs', 'BP', ('ubma','agic','wgic'))
>>> funcsim('path/to/file_ProtIDs_GOIDs', measure=('ubma','agic','wgic'))
	  
FS measure
-----------
measure or tuple of measures under consideration (up to three measures 
can be considered). The symbole of a given functional similarity 
measure is constructed as follows:
The starting letter r, n, l, li, s, x, a, z, w, and u represent GO term
 semantic similarity approaches and stand for Resnik, Nunivers, Lin, 
Li, Relevance, XGraSM, Annotation-based, Zhang, Wang and GO-universal, 
respectively. The suffixes gic, uic, dic, avg, bma, abm and max 
represent SimGIC, SimUIC, SimDIC, Average, Best Match Average, 
Average Best Matches and Max measures, respectively. In cases where the
 prefix x is used, indicating XGraSM-based, it is immediately followed 
by the approach prefix. For example:
    'xlmax' for XGraSM-Lin based Average Functional Similarity Measure
    'agic' for Annotation-based SimGIC Functional Similarity Measure
    'zuic' for Zhang et al. based SimUIC Functional Similarity Measure
And the Union-Intersection functional similarity measure is represented
 by the symbol 'simui'. Refer to the package documentation for details.  
	  
drop : boolean variable only useful in the context of Annotation-based 
approach and it is set to 0 if Inferred from Electronic Annotation 
(IEA) evidence code should be considered and to 1 otherwise. It is set
to 0 by default.
     
output: a boolean variable also set to 1 to output results on the 
screen and to 0 in a file
	     
Example:
--------
	>>> funcsim('tests/TestProteins.txt', measure = ('wcou','acou','acot'))
	>>> a = {'Q5H9L2':['GO:0006355','GO:0006351'], 'P03891':['GO:0022904','GO:0044281','GO:0044237','GO:0006120'], 'Q5H9L2':['GO:0006355','GO:0006351']}
	>>> funcsim(a, measure = ('wvhdf','zvhdf','nvhdf'))
	>>> funcsim(a, ontology = 'BP', measure = 'agic', drop = 1, output=1)
	>>> funcsim(a)
	>>> funcsim(a, measure=('zmax','agic','wgic'))
	>>> A = ['GO:0022904', 'GO:0044281', 'GO:0044237', 'GO:0006120']; B = ['GO:0006355', 'GO:0006351']
	>>> funcsim(A, B, measure = ('ub','nto','db','ub'))
	>>> funcsim('tests/SpecificRefSet1.txt', measure=('zmax','agic','wgic')
	>>> funcsim('tests/SpecificRefSet2.txt', measure = ('wcou','acou','acot', 'ubmm'))
	"""
	print("\n************************************************************************************************************")
	print("       Package A-DaGO-Fun: A General Gene Ontology Semantic Similarity based Functional Analysis Tool")
	print("           Computational Biology Group (CBIO) & African institute for Mathematical Sciences (AIMS)")
	print("                        Distributed under free software (GNU General Public Licence) ")
	print("                             (c) 2015 GPL, Verson 15.1, All rights reserved.")
	print("************************************************************************************************************\n")
	# Defining different variables	
	global Ont, protgo, icc, termcc, TermSim, MaxValue
	Fsim = {'agic':(1,1),'adic':(1,2),'auic':(1,3),'acou':(1,4), 'acot':(1,5), 'xravg':(1,4,10),'xrbma':(1,4,11),'xrabm':(1,4,12),'xrbmm':(1,4,13), 'xrhdf':(1,4,14), 'xrvhdf':(1,4,15), 'xrmax':(1,4,16),'xnavg':(1,5,10),'xnbma':(1,5,11),'xnabm':(1,5,12), 'xnbmm':(1,5,13), 'xnhdf':(1,5,14), 'xnvhdf':(1,5,15),'xnmax':(1,5,16),'xlavg':(1,6,10),'xlbma':(1,6,11),'xlabm':(1,6,12),'xlbmm':(1,6,13), 'xlhdf':(1,6,14), 'xlvhdf':(1,6,15),'xlmax':(1,6,16),'ravg':(1,7,10),'rbma':(1,7,11),'rabm':(1,7,12),'rbmm':(1,7,13), 'rhdf':(1,7,14), 'rvhdf':(1,7,15),'rmax':(1,7,16),'navg':(1,8,10),'nbma':(1,8,11),'nabm':(1,8,12),'nbmm':(1,8,13), 'nhdf':(1,8,14), 'nvhdf':(1,8,15),'nmax':(1,8,16),'lavg':(1,9,10),'lbma':(1,9,11),'labm':(1,9,12),'lbmm':(1,9,13), 'lhdf':(1,9,14), 'lvhdf':(1,9,15),'lmax':(1,9,16),'savg':(1,10,10),'sbma':(1,10,11),'sabm':(1,10,12),'sbmm':(1,10,13), 'shdf':(1,10,14), 'svhdf':(1,10,15),'smax':(1,10,16),'liavg':(1,11,10),'libma':(1,11,12),'liabm':(1,11,12),'libmm':(1,11,13), 'lihdf':(1,11,14), 'livhdf':(1,11,15),'limax':(1,11,16),'ugic':(2,1),'udic':(2,2),'uuic':(2,3),'ucou':(2,4), 'ucot':(2,5),'uavg':(2,1,10),'ubma':(2,1,11),'uabm':(2,1,12),'ubmm':(2,1,13), 'uhdf':(2,1,14), 'uvhdf':(2,1,15),'umax':(2,1,16),'wgic':(3,1),'wdic':(3,2),'wuic':(3,3),'wcou':(3,4), 'wcot':(3,5),'wavg':(3,2,10),'wbma':(3,2,11),'wabm':(3,2,12), 'wbmm':(3,2,13), 'whdf':(3,2,14), 'wvhdf':(3,2,15),'wmax':(3,2,16),'zgic':(4,1),'zdic':(4,2),'zuic':(4,3),'zcou':(4,4), 'zcot':(4,5),'zavg':(4,3,10),'zbma':(4,3,11),'zabm':(4,3,12),'zbmm':(4,3,13), 'zhdf':(4,3,14), 'zvhdf':(4,3,15),'zmax':(4,3,16),'ui':(1,6), 'ub':(1,7), 'db':(1,8), 'nto':(1,9)}
	appnames = {'agic':'Annotation-based SimGIC','adic':'Annotation-based SimDIC','auic':'Annotation-based SimGUIC','xravg':'XGraSM-Resnik based Average','xrbma':'XGraSM-Resnik based Best Match Average','xrabm':'XGraSM-Resnik based Averaging Best Matches','xrmax':'XGraSM-Resnik based Maximum','xnavg':'XGraSM-Nunivers based Average','xnbma':'XGraSM-Nunivers based Best Match Average','xnabm':'XGraSM-Nunivers based Averaging Best Matches','xnmax':'XGraSM-Nunivers based Maximum','xlavg':'XGraSM-Lin based Average','xlbma':'XGraSM-Lin based Best Match Average','xlabm':'XGraSM-Lin based Averaging Best Matches','xlmax':'XGraSM-Lin based Maximum','ravg':'Resnik-based Average','rbma':'Resnik-based Best Match Average','rabm':'Resnik-based Averaging Best Matches','rmax':'Resnik-based Average','navg':'Nunivers-based Average','nbma':'Nunivers-based Best Match Average','nabm':'Nunivers-based Averaging Best Matches','nmax':'Nunivers-based Maximum','lavg':'Lin-based Average','lbma':'Lin-based Best Match Average','labm':'Lin-based Averaging Best Matches','lmax':'Lin-based Maximum','savg':'Relevance-based Average','sbma':'Relevance-based Best Match Average','sabm':'Relevance-based Averaging Best Matches','smax':'Relevance-based Maximum','liavg':'Li-based Average','libma':'Li-based Best Match Average','liabm':'Li-based Averaging Best Matches','limax':'Li-based Maximum','ugic':'GO-universal based SimGIC','udic':'GO-universal based SimDIC','uuic':'GO-universal based SimGIC','uavg':'GO-universal based Average','ubma':'GO-universal based Best Match Average','uabm':'GO-universal based Averaging Best Matches','umax':'GO-universal based Maximum','wgic':'Wang et al. based SimGIC','wdic':'Wang et al. based SimDIC','wuic':'Wang et al. based SimGUIC','wavg':'Wang et al. based Average','wbma':'Wang et al. based Best Match Average','wabm':'Wang et al. based Averaging Best Matches','wmax':'Wang et al. based Maximum','zgic':'Zhang et al. based SimGIC','zdic':'Zhang et al. based SimDIC','zuic':'Zhang et al. based SimGUIC','zavg':'Zhang et al. based Average','zbma':'Zhang et al. based Best Match Average','zabm':'Zhang et al. based Averaging Best Matches','zmax':'Zhang et al. based Maximum','ui':'Union-Intersection based','rbmm':'Resnik based Best Match Maximim','rhdf':'Resnik-based derived from the Hausdorff metric','rvhdf':'Resnik-based derived from a variant Hausdorff measure','xrbmm':'XGraSM-Resnik based Best Match Maximum','xrhdf':'XGraSM-Resnik-based derived from the Hausdorff metric','xrvhdf':'XGraSM-Resnik-based derived from a Variant Hausdorff measure','xnbmm':'XGraSM-Nunivers based Best Match Maximim','xnhdf':'XGraSM-Nuvivers-based derived from the Hausdorff metric','xnvhdf':'XGraSM-Nunivers-based derived from a Variant Hausdorff measure','nbmm':'NUnivers based Best Match Maximum','nhdf':'NUnivers-based derived from the Hausdorff metric','nvhdf':'Nunivers-based derived from a Variant Hausdorff measure','lbmm':'Lin based Best Match Maximum','lhdf':'Lin-based derived from the Hausdorff metric','lvhdf':'Lin-based derived from a variant Hausdorff','xlbmm':'XGraSM-Lin based Best Match Maximim','xlhdf':'XGraSM-Lin-based derived from the Hausdorff metric','xlvhdf':'XGraSM-Lin-based derived from a variant Hausdorff measure','sbmm':'Relevance based Best Match Maximum','shdf':'Relevance-based derived from the Hausdorff metric','svhdf':'Relevance-based derived from a variant Hausdorff measure','lihdf':'Li et al.-based derived from the Hausdorff metric','livhdf':'Li et al.-based derived from a variant Hausdorff measure','libmm':'Li et al. based Best Match Maximum','acou':'Annotation-based SimCOU','acot':'Annotation-based SimCOT','ubmm':'GO-Universal based Best Match Maximum','uhdf':'GO-universal based functional similarity derived from the Hausdorff metric','uvhdf':'GO-universal based functional similarity derived from a variant Hausdorff measure','ucou':'GO-universal-based SimCOU','ucot':'GO-universal-based SimCOT','wbmm':'Wang et al. based Best Match Maximum','whdf':'Wang et al. based functional similarity derived from the Hausdorff metric','wvhdf':'Wang et al. based functional similarity derived from a variant Hausdorff measure','wcou': 'Wang et al. based SimCOU','wcot':'Wang et al. based SimCOT','zbmm':'Zhang et al. based Best Match Maximum','zhdf':'Zhang et al. based functional similarity derived from the Hausdorff metric','zvhdf':'Zhang et al. based functional similarity derived from a variant Hausdorff measure','zcou':'Zhang et al. based SimCOU','zcot':'Zhang et al. based SimCOT','ub':'Universal-based','db':'Dice-based','nto':'Normalized Term Overlap based','ub':'Universal-based'}
	sdpt = -1 # Indicate the source of data: from a dictionary read 1 or from a file 0 and 2 for a two set GO IDs.

	now = time.time()
	print("Calculating Functional similarity scores on %s\n"%str(time.asctime(time.localtime())))
	
	#print "    Checking different parameters ..."
	if len(args) < 1:
		print('Illegal number of arguments: at least one argument required, no argument given.\nFor now, the process cannot be pursued: number of arguments error ...')
		sys.exit(1)
	elif  len(args)+len(kwargs) > 6:
		print('Illegal number of arguments: at most 6 arguments required, more than 6 arguments given.\nFor now, the process cannot be pursued: number of arguments error ...')
		sys.exit(2)
	elif len(args)==1: # This means that the only argument is a file containing GO ID pairs
		if type(args[0])==dict: sdpt = 1
		elif os.path.exists(os.path.join(os.path.dirname(__file__), args[0])): sdpt = 0
		else: # raise IOError
			print("The file or object provided cannot be opened for reading.\nThere is argument value error. Check and try again ...")
			sys.exit(3)
		kwargs = _fixkwargs(kwargs, Fsim)
	elif 2<=len(args)<=6: # This means that arguments are protein/gene pairs or a file/dictionary and ontology
		if isinstance(args[0], (tuple, list, set)) and isinstance(args[1], (tuple, list, set)): # Two are lists, tuples or sets of GO IDs
			sdpt = 2
		elif type(args[0])==dict:
			sdpt, kwargs['Targets'] = 1, args[1]
		elif os.path.exists(os.path.join(os.path.dirname(__file__), args[0])):
			sdpt, kwargs['Targets'] = 0, args[1]
		else: #raise IOError
			print("The file or object provided cannot be opened for reading.\nThere is argument value error. Check and try again ...")
			sys.exit(4)
		if len(args)>=3: 
			kwargs['ontology'] = args[2]
			if len(args)>=4: 
				kwargs['measure'] = args[3]
				if len(args)>=5: 
					kwargs['drop'] = args[4]
					if len(args)>=6: kwargs['output'] = args[5]
		kwargs = _fixkwargs(kwargs, Fsim)
	else:
		print("There is inconsistency in your input parameters. Please check different parameters provided and try again ...") 
		sys.exit(5)
	
	# Loading GO data
	sontology = 1 + Ont.index(kwargs['ontology'])
	getGOdata(sontology)

	# Loading Protein and GO terms
	protgo.clear()
	if not sdpt: readProteinFile(args[0])
	elif sdpt==1: # Dealing with a dictionary with protein as key and set/tuple/list of GO IDs as value
		for p in args[0]: 
			protgo[p] = set([icc[t] for t in args[0][p] if (t in icc) and termcc[icc[t]][-1] and icc[t]!=Head])
	elif sdpt==2:
		protgo['TermSet1'] = set([icc[t] for t in args[0] if (t in icc) and termcc[icc[t]][-1] and icc[t]!=Head])
		if not protgo['TermSet1']:
			print("No GOID identified in the first list/tuple/set provided.\nFor now, the process cannot be pursued: Input value error ...")
			sys.exit(6)
		protgo['TermSet2'] = set([icc[t] for t in args[1] if (t in icc) and termcc[icc[t]][-1] and icc[t]!=Head])
		if not protgo['TermSet2']:
			print("In the current GO dataset used, no GOID identified in the second list/tuple/set provided.\nFor now, the process cannot be pursued: Input value error ...\n")
			sys.exit(7)

	# Retrieve protein pairs
	ppairs = []
	if protgo:
		if kwargs['Targets']: # protein/gene pairs provided
			if isinstance(kwargs['Targets'], (list, tuple)):
				ppairs = [tuple(a) for a in kwargs['Targets'] if len(tuple(a))==2]
				ppairs = [a for a in ppairs if a[0] in protgo and a[1] in protgo]
			else:             # Meaning that protein pairs are in the file
				ppairs = inputpairs(kwargs['Targets'])
		else:                 # protein/gene pairs are not provided
			pairs = list(protgo.keys())
			ppairs = [(pairs[i],pairs[j]) for i in range(len(pairs)) for j in range(i+1,len(pairs))]
	else:
		print("Annotation inputs are inconsistent, please check them and try again\nFor now, the process cannot be pursued: Annotation inputs inconsistent ...") 
		sys.exit(8)
	
	if not ppairs:
		print("Protein/gene pairs seem to be inconsistent, please check them and try again\nFor now, the process cannot be pursued: Annotation inputs inconsistent ...\n") 
		sys.exit(9)

	apprsymb = {}; appr = []
	for a in kwargs['measure']:
		apprsymb[Fsim[a]] = a; appr.append(Fsim[a])
	appr.sort()
	data = {}; mapp = 0
	for i in range(len(appr)):
		if appr[i][0] != mapp:
			getICdata(sontology, appr[i][0], kwargs['drop'])
			mapp = appr[i][0]
		if len(appr[i])==3:      	# Corresponding to the term pair based protein functional similarity 
			Tmp = {}
			if (appr[i][1]==4 or appr[i][1]==7) and not MaxValue: MaxValue = max(goic.values())
			for p in ppairs:
				if p[0] in protgo and p[1] in protgo:
					retrieveTermSimilarity(p, appr[i][1])
					Tmp[p] = computingProteinSimilarityScore(p, appr[i][-1])
			data[apprsymb[appr[i]]] = Tmp.copy()
			del Tmp
		elif len(appr[i])==2:
			Tmp = {}
			for p in ppairs:
				if p[0] in protgo and p[1] in protgo: Tmp[p] = computingProteinSimilarityScore(p, appr[i][-1])
			data[apprsymb[appr[i]]] = Tmp.copy()
			del Tmp
	
	outs = []
	for p in ppairs:
		outs.append([p[0],p[1]])
		for a in kwargs['measure']:
			if p in data[a]: outs[-1].append(data[a][p]) # Checking first if p is in data[p] as one of the proteins/genes can be unchar.
			else: outs[-1].append('U')
	# Returning a Python object: a list of tuples containing pairwise proteins and associated semantic similarity scores
	if kwargs['output']==2:
		print("\nCalculating protein pairwise functional similarity scores accomplished on %s"%str(time.asctime(time.localtime())))
		print("Total time elapsed is approximately:", (time.time()-now), 'seconds\n')
		print("\n************************************************************************************************************\n")
		return outs 
	# Outputting different results
	print("Computing functional similarity scores using  :", "[funcsim function from the module ProteinSimilarity.py]")
	print("Number of possible protein/gene pairs detected:", len(outs))
	print("Different functional similarity measures      :", ", ".join([a.upper() for a in kwargs['measure']]))
	if kwargs['output']:
		print("\nFunctional similarity scores are displayed in the table below.\nIf possible, use full screen mode for more convenient visualization:")
	else:
		inputdata = inspect.getframeinfo(inspect.currentframe().f_back)[3][0].strip().split(',')[0].split('(')[1].strip()
		outputfile = inputdata.replace('\'','').replace('\"','').replace(':','_').split('.')[0].split('/')[-1].strip() + 'FS.txt'
		print("Functional similarity scores can be found in the file: [%s]"%(outputfile,))

	headers = ['Concept-1', 'Concept-2']+[a.upper() for a in kwargs['measure']]
	if kwargs['output']: # Print on the screen
		print(tab(outs, headers, kwargs['tablefmt'], floatfmt=".5f", stralign="center"))
		print("\nLegend of Functional Similarity Measure:")
		print("----------------------------------------")
		for a in kwargs['measure']:
			print("  %s: %s Functional Similarity Measure"%(a.upper(), appnames[a]))
		print("\n  Note: Possible 'U' as score indicates that at GO annotations of at least one proteins cannot be\n\ttranslated into the current GO used.")
	else:
		try:
			fp = open(outputfile, 'w')
			fp.write("# Computing functional similarity for following measure(s):\n")
			for a in kwargs['measure']:
				fp.write("# %s: %s functional similarity measure\n"%(a.upper(), appnames[a]))
			fp.write("# Note: Possible 'U' as score indicates that at GO annotations of at least one\n#\t\tproteins cannot be translated into the current GO used.\n\n")
			fp.write('%s'%tab(outs, headers, kwargs['tablefmt'], floatfmt=".5f", stralign="center"))
			fp.close()
		except IOError:
			print("File cannot be opened in writing. Check possibly the writing permission and try again ...")
			sys.exit(8)
	print("\nProcessing accomplished on %s"%str(time.asctime(time.localtime())))
	print("Total time elapsed is approximately", (time.time()-now), 'seconds') 
	print("\n************************************************************************************************************\n")

def _main(argv):
	"""
This constitutes a useful function for a user who chooses not to use 
the Python interpreter, but to run the package using a bash command
line. Note that this is only applicable in the case where user 
proteins' input data retrieved from files. In this case, retrieving 
terms' features is achieved using the following command:

	python $(python -m site --user-site)/dagofun/ProteinSimilarity.py AnnotationFile ProteinPairs ontology nmeas measure drop output

Different arguments are as explained in the termsim function (see pa-
ckage documentation). Except that nmeas which is the new argument and
represents the number of measures to be executed as the module can si-
multaneously compute protein functional similarity scores using more 
than one measure and can go up to four different measures. These mea-
sures are then provided just after providing this number.
Moreover these arguments should be in order as shown in the command and 
as for commands under a Python interpreter, AnnotationFile file con-
taining the user input list of proteins and their associated GO IDs and 
ProteinPairs is another user input file containing protein pairs for 
which scores should be computed, and these two files must be provided. 
In case where other parameters are not provided, default parameters are 
used.

Assuming that the package was not installed, then:

$(python -m site --user-site)/dagofun/

should be replaced by the path to modules.
	"""
	if len(argv) <= 2:
		print('\nIllegal number of arguments: at least one argument required, no argument given.\nFor now, the process cannot be pursued: number of arguments error ...\n')
	elif  len(argv) > 11:
		print('\nIllegal number of arguments: at most 10 argument required, more than 4 arguments given.\nFor now, the process cannot be pursued: number of arguments error ...\n')
	else:
		if len(argv)==3: funcsim(argv[1].strip(), argv[2].strip())
		elif len(argv)==4: funcsim(argv[1].strip(), argv[2].strip(), argv[3].strip())
		else: 
			try:
				nmeas = int(argv[4])
				meas = []; j = 5
				for i in range(nmeas):
					try:
						tmeas = argv[5+i].strip(); j += 1
						if not tmeas in meas: meas.append(tmeas)
					except:
						print("\nThere is inconsistency between the number of measures provided and expected \nnumber of measures. Please refer to the package documentation,\ncheck nmeas and measures provided, and try again ...\n")
						return
				Drop = 0; Output = 1
				if len(argv) > j:
					try:
						Drop = int(argv[j].strip()); j += 1
						if len(argv) > j: Output = int(argv[j].strip())
					except:
						print("\nThere is inconsistency. Please refer to the package documentation,\ncheck drop and/or output parameters provided ...\n")
						return 
				if Output != 1 and Output != 0:
					print ("\nHow do you want to output results should be an Enum 0, 1, 2:\n\t1 if results should be displayed on the screen \n\t0 if results should be written in a file.\n\t2 for outputting a Python object for possible further usage.\n\nRunning this module using a command line is plausible only when\nthe output parameter is set to 0 or 1.\nPlease check and try again ...\n")
					return
				if nmeas != len(meas):
					print("\nWarning:: Please note that the number of approaches provided and the actual number of approaches are different.\nThe list of approaches provided is redundant.\n")
				funcsim(argv[1].strip(), argv[2].strip(), ontology = argv[3].strip(), measure = tuple(meas), drop = Drop, output = Output)
			except:
				print("\nThere is inconsistency. Please refer to the package documentation,\ncheck the nmeas argument provided and try again ...\n")

if __name__=='__main__':
	_main(sys.argv)
	
