#!/usr/bin/python

"""
This python file is part of the DaGO-Fun tool, which is a tool for Gene 
Ontology-based functional analysis using term information content 
measures.
This python code implements enrichment analysis where the term dependen-
cies in the GO DAG and and the uncertainty in annotation data using 
fuzzy expressions through semantic similarity concepts. Furthemore, this
is context independent and can be used for any GO annotated dataset as
population background or reference and not any fi.

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
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,  * WHETHER IN AN 
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
See <http://www.gnu.org/licenses/>.

This code was written by 
    Gaston K. Mazandu <gmazandu@{cbio.uct.ac.za, gmail.com}, 
                       kuzamunu@aims.ac.za>
    (c) 2015 under free software (GPL), all rights reserved.
"""

__all__ = ["gossfeat"]
__version__= "15.1"
__author__ = """Gaston K. Mazandu <gmazandu@{cbio.uct.ac.za, gmail.com}, kuzamunu@aims.ac.za>"""

# Importing necessary python libraries
import sys, os, re, inspect, time

# Importing external python modules and package
from tabulate import tabulate as tab
from functools import reduce

try: # It is sufficient to import "cPickle" here once for reading or storing python binary files
	import pickle 
except ImportError:
    import pickle as cPickle

try: # It is sufficient to import "scipy" here once for some useful mathematical objects, functions and algorithm needed.
	from scipy import log, exp, sort, arange, array, mean, ceil
	import scipy.stats as dst
except ImportError:
	raise ImportError("The library scipy is required for some math objects. \nPlease, install it and try again ...")

# Defining necessary global variables
goic, lcc, gocf, termcc, icc, ccind, protgo, protargets = {}, {}, {}, {}, {}, {}, {}, set()
Head = None
Fam = ['AnnChar', 'Universal', 'WangIC', 'Zhang']
Ont = ['BP', 'MF', 'CC']

# Function definitions start here!
def getGOdata(so):
	"""
		For loading all gene ontology features for the ontology under consideration.
	"""
	global Head, lcc, Ont, icc, termcc, ccind
	lcc =  pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sLevelAncestor.ck'%(Ont[so-1], )),'rb'))
	icc = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sTermIndex.ck'%(Ont[so-1], )),'rb'))
	ccind = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sIndexTerms.ck'%(Ont[so-1],)),'rb'))
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
	if sf==1 and drop==1:
		goic = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%s%sPartialI.ck'%(Ont[so-1], Fam[sf-1])),'rb'))
	else:
		goic = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%s%sI.ck'%(Ont[so-1], Fam[sf-1])),'rb'))
	
	if sf==3: 
		gocf = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sWangCFI.ck'%(Ont[so-1],)),'rb'))
	return

def Sort(x,y):
	if (x[1] < y[1]): return 1
	else: return -1

def simProt(targetterms, targetproteins, background, app, agree):
	""" 
		Inputs: takes 5 parameters: the set of terms in the target 'targetterms', set of protein target 'targetproteins', set of proteins in the background proteins 'background', Approach under consideration 'app' and Agreement level 'agree'.
		 Outputs: returns fuzzy frequency of occurrences of each term in the target and reference gene sets given term semantic similarity approach under consideration.
	"""
	global goic, lcc, gocf, Head, protgo
	nt = len(targetterms); stringp = "Processing : %s %d%s"%(100*'.',0,'%'); count = 0
	getprot = {}
	if app==1: # GO-universal 
		for t in targetterms:
			count += 1
			print(stringp, end=' ')
			initset = set(lcc[t][1])
			l = 0; m = 0
			for p in background:   # Fuzzy counting the number of times the term occurs in the back and target sets
				if t in protgo[p]: # The term occurs in p
					m += 1
					if p in targetproteins: l += 1
				else:
					for s in protgo[p]:     # Search for fuzzy occurrence
						csimt = 0.0
						A = set(lcc[s][1])
						if t in A: # This means that t is an ancestor of s occurring
							csimt = goic[t]/goic[s]
							if csimt >= agree:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
						elif s in initset:   # This means that t is a child of s occurring
							csimt = goic[s]/goic[t]
							if csimt >= 0.7:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
			perc = (count*100)//nt
			stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
			getprot[t] = (l, m)
		print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')
	elif app==2: # This is Wang et al. approach
		for t in targetterms:
			count += 1
			print(stringp, end=' ')
			initset = set(gocf[t].keys()) # set(lcc[t][1]+[t])
			l = 0; m = 0
			for p in background:
				if t in protgo[p]: # The term occurs in p
					m += 1
					if p in targetproteins: l += 1
				else:
					for s in protgo[p]:
						csimt = 0.0
						A = set(gocf[s].keys())
						if t in A:  # This means that t is an ancestor of s occurring
							csimt = sum([gocf[t][a]+gocf[s][a] for a in initset])/(goic[t]+goic[s])
							if csimt >= agree:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
						elif s in initset:   # This means that t is a child of s occurring
							csimt = sum([gocf[t][a]+gocf[s][a] for a in A])/(goic[t]+goic[s])
							if csimt >= 0.7:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
			perc = (count*100)//nt
			stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
			getprot[t] = (l, m)
		print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')
	elif app==3: # This is Zhang et al. approach
		for t in targetterms:
			count += 1
			print(stringp, end=' ')
			initset = set(lcc[t][1])
			l = 0; m = 0
			for p in background:   # Fuzzy counting the number of times the term occurs in the back and target sets
				if t in protgo[p]: # The term occurs in p
					m += 1
					if p in targetproteins: l += 1
				else:
					for s in protgo[p]:     # Search for fuzzy occurrence
						csimt = 0.0
						A = set(lcc[s][1])
						if t in A: # This means that t is an ancestor of s occurring
							csimt = 2.0*goic[t]/(goic[s]+goic[t])
							if csimt >= agree:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
						elif s in initset:   # This means that t is a child of s occurring
							csimt = 2.0*goic[s]/(goic[s]+goic[t])
							if csimt >= 0.7:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
			perc = (count*100)//nt
			stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
			getprot[t] = (l, m)
		print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')
	elif app==4: # XGraSM-Resnik
		MaxValue = max(goic.values())
		for t in targetterms:
			count += 1
			if not t in goic: continue # This is an annotation-based approach
			print(stringp, end=' ')
			initset = set(lcc[t][1]+[t])
			initset.discard(Head)
			if not initset: continue
			l = 0; m = 0
			for p in background:
				if t in protgo[p]: # The term occurs in p
					m += 1
					if p in targetproteins: l += 1
				else:
					for s in protgo[p]:
						csimt = 0.0
						A = set(lcc[s][1]+[s])
						A.discard(Head)
						if not A: continue
						if t in A: 
							csimt = mean([goic[a] for a in initset])/MaxValue
							if csimt >= agree:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
						elif s in initset:   # This means that t is a child of s occurring
							csimt = mean([goic[a] for a in A])/MaxValue
							if csimt >= 0.7:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
			perc = (count*100)//nt
			stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
			getprot[t] = (l, m)
		print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')
	elif app==5: # XGraSM-Nunivers
		for t in targetterms:
			count += 1
			if not t in goic: continue # This is an annotation-based approach
			print(stringp, end=' ')
			initset = set(lcc[t][1]+[t])
			initset.discard(Head)
			if not initset: continue
			l = 0; m = 0
			for p in background:
				if t in protgo[p]: # The term occurs in p
					m += 1
					if p in targetproteins: l += 1
				else:
					for s in protgo[p]:
						csimt = 0.0
						A = set(lcc[s][1]+[s])
						A.discard(Head)
						if not A: continue
						if t in A: 
							csimt = mean([goic[a] for a in initset])/goic[s]
							if csimt >= agree:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
						elif s in initset:   # This means that t is a child of s occurring
							csimt = mean([goic[a] for a in A])/goic[t]
							if csimt >= 0.7:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
			perc = (count*100)//nt
			stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
			getprot[t] = (l, m)
		print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')
	elif app==6: # XGraSM-Lin
		for t in targetterms:
			count += 1
			if not t in goic: continue # This is an annotation-based approach
			print(stringp, end=' ')
			initset = set(lcc[t][1]+[t])
			initset.discard(Head)
			if not initset: continue
			l = 0; m = 0
			for p in background:
				if t in protgo[p]: # The term occurs in p
					m += 1
					if p in targetproteins: l += 1
				else:
					for s in protgo[p]:
						csimt = 0.0
						A = set(lcc[s][1]+[s])
						A.discard(Head)
						if not A: continue
						if t in A: 
							csimt = 2.0*mean([goic[a] for a in initset])/(goic[s]+goic[t])
							if csimt >= agree:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
						elif s in initset:   # This means that t is a child of s occurring
							csimt = 2.0*mean([goic[a] for a in A])/(goic[s]+goic[t])
							if csimt >= 0.7:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
			perc = (count*100)//nt
			stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
			getprot[t] = (l, m)
		print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')
	elif app==7: # Resnik
		MaxValue = max(goic.values())
		for t in targetterms:
			count += 1
			if not t in goic: continue # This is an annotation-based approach
			print(stringp, end=' ')
			initset = set(lcc[t][1])
			l = 0; m = 0
			for p in background:
				if t in protgo[p]: # The term occurs in p
					m += 1
					if p in targetproteins: l += 1
				else:
					for s in protgo[p]:
						csimt = 0.0
						A = set(lcc[s][1])
						if t in A: 
							csimt = goic[t]/MaxValue
							if csimt >= agree:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
						elif s in initset:   # This means that t is a child of s occurring
							csimt = goic[s]/MaxValue
							if csimt >= 0.7:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
			perc = (count*100)//nt
			stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
			getprot[t] = (l, m)
		print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')	
	elif app==8: # Nunivers
		for t in targetterms:
			count += 1
			if not t in goic: continue # This is an annotation-based approach
			print(stringp, end=' ')
			initset = set(lcc[t][1]+[t])
			l = 0; m = 0
			for p in background:
				if t in protgo[p]: # The term occurs in p
					m += 1
					if p in targetproteins: l += 1
				else:
					for s in protgo[p]:
						csimt = 0.0
						A = set(lcc[s][1]+[s])
						if t in A: 
							csimt = goic[t]/goic[s]
							if csimt >= agree:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
						elif s in initset:    # This means that t is a child of s occurring
							csimt = goic[s]/goic[t]
							if csimt >= 0.7:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
			perc = (count*100)//nt
			stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
			getprot[t] = (l, m)
		print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')
	elif app==9: # Lin
		for t in targetterms:
			count += 1
			if not t in goic: continue # This is an annotation-based approach
			print(stringp, end=' ')
			initset = set(lcc[t][1]+[t])
			l = 0; m = 0
			for p in background:
				if t in protgo[p]: # The term occurs in p
					m += 1
					if p in targetproteins: l += 1
				else:
					for s in protgo[p]:
						csimt = 0.0
						A = set(lcc[s][1]+[s])
						if t in A: 
							csimt = 2.0*goic[t]/(goic[t]+goic[s])
							if csimt >= agree:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
						elif s in initset:    # This means that t is a child of s occurring
							csimt = 2.0*goic[s]/(goic[t]+goic[s])
							if csimt >= 0.7:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
			perc = (count*100)//nt
			stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
			getprot[t] = (l, m)
		print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')
	elif app==10: # Relevance
		for t in targetterms:
			count += 1
			if not t in goic: continue # This is an annotation-based approach
			print(stringp, end=' ')
			initset = set(lcc[t][1]+[t])
			l = 0; m = 0
			for p in background:
				if t in protgo[p]: # The term occurs in p
					m += 1
					if p in targetproteins: l += 1
				else:
					for s in protgo[p]:
						csimt = 0.0
						A = set(lcc[s][1]+[s])
						if t in A: 
							csimt = 2.0*goic[t]*(1.0-exp(-goic[t]))/(goic[t]+goic[s])
							if csimt >= agree:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
						elif s in initset:    # This means that t is a child of s occurring
							csimt = 2.0*goic[s]*(1.0-exp(-goic[s]))/(goic[t]+goic[s])
							if csimt >= 0.7:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
			perc = (count*100)//nt
			stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
			getprot[t] = (l, m)
		print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')
	elif app==11: # Li et al. (SimIC)
		for t in targetterms:
			count += 1
			if not t in goic: continue # This is an annotation-based approach
			print(stringp, end=' ')
			initset = set(lcc[t][1]+[t])
			l = 0; m = 0
			for p in background:
				if t in protgo[p]: # The term occurs in p
					m += 1
					if p in targetproteins: l += 1
				else:
					for s in protgo[p]:
						csimt = 0.0
						A = set(lcc[s][1]+[s])
						if t in A: 
							csimt = 2.0*goic[t]*(1.0-1.0/(1.0+goic[t]))/(goic[t]+goic[s])
							if csimt >= agree:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
						elif s in initset:    # This means that t is a child of s occurring
							csimt = 2.0*goic[s]*(1.0-1.0/(1.0+goic[s]))/(goic[t]+goic[s])
							if csimt >= 0.7:  # Indicating that the term fuzzy-occurs in the protein p
								m += 1
								if p in targetproteins: l += 1
								break
			perc = (count*100)//nt
			stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
			getprot[t] = (l, m)
		print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')
	return getprot

def simProt01(targetterms, targetproteins, background, agree):
	""" 
		Inputs: takes 5 parameters: the set of terms in the target 'targetterms', set of protein target 'targetproteins', set of proteins in the background proteins 'background', Approach under consideration 'app' and Agreement level 'agree'.
		 Outputs: returns frequency of occurrences of each term in the target and reference gene sets given term semantic similarity approach under consideration.
	"""
	global goic, lcc, gocf, Head, protgo
	nt = len(targetterms); stringp = "Processing : %s %d%s"%(100*'.',0,'%'); count = 0
	getprot = {}
	for t in targetterms:
		count += 1
		print(stringp, end=' ')
		initset = set(lcc[t][1]+[t])
		l = 0; m = 0
		for p in background:
			if t in protgo[p]:     # The term occurs in p
				m += 1
				if p in targetproteins: l += 1
			elif agree==0.0:       # Get here only when considering true-path rule when agree==0
				for s in protgo[p]:
					A = set(lcc[s][1]+[s])
					if t in A:     # Indicating that the term t occurs as an ancestor of s
						m += 1
						if p in targetproteins: l += 1
						break
		perc = (count*100)//nt
		stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
		getprot[t] = (l, m)
	print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')		
	return getprot

def filterSet(termset):
	"""
		This module 
	"""
	global goic, lcc, gocf, Head
	fl03 = {}; setvisited = set()
	for i in range(len(termset)):
		if termset[i] in setvisited: continue
		fl03[termset[i]] = set(); t = termset[i]
		for j in range(i+1, len(termset)):
			if termset[j] in lcc[t][1]:
				fl03[t].add(termset[j])
				setvisited.add(termset[j])
				try: # This j as ancestor should be removed and all its parents added to i!
					fl03[t] |= fl03[termset[j]]					
					fl03.pop(termset[j], None)
				except:
					pass
	return fl03

def readbackground(FileName):
	""" 
		Reading protein file and constructing ProteinGO dictionary from the database, it takes 3 parameters and return 2 values:
		Inputs: FileName is the name of file containing protein and GO annotations. The file has two columns separated by while space, the first column is the protein ID and the second column is the set of GO terms associated to the protein and these terms must be separated by a comma.
		Outputs: It returns a dictionary with protein ID as keys having values a set of indexes associated to its set of GO terms. 
	"""
	global icc, termcc, Head, protgo
	protgo.clear()
		
	try:
		fp = open(os.path.join(os.path.dirname(__file__), FileName),'r')
	except:
		print("Check the path to your file or if the file exists ...")
		sys.exit(0)

	for line in fp:
		ligne = line.strip()
		if not ligne: continue
		ligne = ligne.split()
		if len(ligne)!=2: continue
		protein = ligne[0].strip()
		termprotein = set([x.strip() for x in ligne[1].split(',')])
		protgo[protein] = set([icc[t] for t in termprotein if (t in icc) and termcc[icc[t]][-1] and icc[t]!=Head])
		if not protgo[protein]: protgo.pop(protein, None) # Delete safely the concept added when it does not have annotations
		del termprotein
	fp.close()

def readtargets(FileName):
	""" 
		Reading protein file and constructing ProteinGO dictionary from the database, it takes 3 parameters and return 2 values:
		Inputs: FileName is the name of file containing protein pairs, Id is the protein Identifiers 1 for UniProt Ids and 2 for Genenames
				  and so indicating the ontology undeer consideration!
		Outputs: 
	"""
	global protargets
	protargets.clear()
	try:
		fp = open(os.path.join(os.path.dirname(__file__), FileName), 'r')
	except:
		print("Check the path to your file or if the file exists ...")
		sys.exit(0)
	for line in fp:
		ligne = line.strip()
		if not ligne: continue
		ligne = ligne.split()
		protein = ligne[0].strip()
		protargets.add(protein)
		del protein
	fp.close()

def _fixkwargs(dicts, Alapp):
	"""
		This module checks different parameters *args and **kwargs and align them correctly in order to run termsim module.
	"""
	if len(dicts) > 8:
		print("8 arguments are required but more than 9 arguments provided.\n") #raise IOError
		sys.exit(1)
	
	kwargs = {}
	if 'ontology' not in dicts: kwargs['ontology'] = 'BP'
	elif dicts['ontology'].upper() in ['BP','MF','CC']: kwargs['ontology'] = dicts['ontology'].upper()
	else:
		print("Check your ontology, the TermSimilarity module uses the string:\n\t'BP': For Biological Process\n\t'MF': For Molecular Function\n\t'CC': For Cellular Component.\nunknown ontology was provided") # raise ValueError
		sys.exit(2)

	if 'approach' not in dicts: kwargs['approach'] = 'u'
	elif type(dicts['approach'])==str: 
		if dicts['approach'].lower() in Alapp: kwargs['approach'] = dicts['approach'].lower()
		else:
			print("Check notations of different approaches from the package documentation.\n\nFor now, the process cannot be pursued: approach key error ...\n") # raise ValueError
			sys.exit(3)
	else:
		print("Approach key should be a symbol of approach. Please check and try again ...\n") # raise TypeError
		sys.exit(4)

	if 'score' not in dicts: kwargs['score'] = 0.3
	elif isinstance(dicts['score'], (int, float)) and 0.0 <= dicts['score'] <= 1.0: kwargs['score'] = float(dicts['score'])
	else:
		print("Check your semantic similarity score cut-off, it must be a real number: 0 < score <= 1.\n\nFor now, the process cannot be pursued: Cut-off score value error ...\n")
		sys.exit(5)

	if 'pvalue' not in dicts: kwargs['pvalue'] = 0.05
	elif type(dicts['pvalue'])==float and 0.0 < dicts['pvalue'] <= 0.05: kwargs['pvalue'] = float(dicts['pvalue'])
	else:
		print("Check your level of significance cut-off, it must be a real number: 0 < pvalue <= 0.05.\n\nFor now, the process cannot be pursued: Cut-off pvalue error ...\n")
		sys.exit(6)

	if 'drop' not in dicts: kwargs['drop'] = 0
	elif dicts['drop'] is 0 or dicts['drop'] is 1: kwargs['drop'] = dicts['drop']
	else:
		print("Check the use of IEA evidence code variable <drop> which should be a Boolean:\n\t0 if all evidence code should be used and \n\t1 if IEA evidence code should be excluded.\n\nPlease check and try again ...\n") # raise ValueError
		sys.exit(7)
	if 'output' not in dicts: kwargs['output'] = 0
	elif dicts['output'] is 0 or dicts['output'] is 2: kwargs['output'] = dicts['output']
	else:
		print("How do you want to output results is an Enum 0, 2:\n\t0 if results should be written in a file.\n\t2 for outputting a Python object for possible further usage.\n\nPlease check and try again ...") # raise ValueError
		sys.exit(7)
	kwargs['tablefmt'] = 'rst' # One can also use 'grid' or other display formats!
	return kwargs

def gossfeat(*args, **kwargs):
	"""
This function retrieves biological processes most pertinent to 
the experiment performed based on the target set and background 
provided. It incorporates the complex dependence structure of 
the GO DAG and the uncertainty in annotation data using fuzzy 
expressions through GO term semantic similarity measures.

*args* is a variable length argument, which is two strings 
representing the name of the target protein file and the file 
of background proteins, each with its GO ID annotations.
		
*kwargs* can be used to set ontology, approach, score and drop 
arguments. 
    	 
  >>> gossfeat(ReferenceFile, TargetFile, ontology='BP', approach='u', score=0.3, pvalue=0.05, drop=0)

* ontology: note that we are dealing with three ontology indepen-
  dently. Biological Process ('BP'), Molecular Function ('MF') and
  Cellular Component (CC). If no ontology is given then 'BP' is used

SS Approach
-----------
Approach or tuple of approaches under consideration:
	'u': for the GO-Universal
	'w': for Wang et al.
	'z': for Zhang et al
	'r': for Resnik
	'xr': for XGraSM-Resnik
	'n': for Nunivers
	'xn': for XGraSM-Nunivers
	'l':  for Lin
	'xl': for XGraSM-Lin
	'li': for Lin enhancement by Li et al.
	's': for Lin enhancement by Relevance (Schlicker et al.)   

* drop : boolean variable only useful in the context of Annotation-
based approach and it is set to 0 if Inferred from Electronic Anno-
tation (IEA) evidence code should be considered and to 1 otherwise.
 By default, it is set to 0.
A default parameters are GO-universal ('u') for SS, score = 0.3 the 
threshold or agreement level.  The significance level cut-off from 
which an identified term is considered to be statistically signifi-
cant, set to 0.05 by default.

(4.2) Examples:
      --------
  >>> from dagofun.EnrichmentAnalysis import *
  >>> gossfeat('tests/ReferenceSetTest.txt', "tests/TargetSetTest.txt", approach = 's', score=0.1)
  >>> gossfeat('tests/ReferenceSetTest.txt', "tests/TargetSetTest.txt")

Note that for this specific function, resulting list all enriched 
terms and their features are displayed in a file from the forder 
where the module is being executed.
	"""
	print("\n*********************************************************************************************************************")
	print("            Package G-DaGO-Fun: A General Gene Ontology Semantic Similarity based Functional Analysis Tool")
	print("                Computational Biology Group (CBIO) & African institute for Mathematical Sciences (AIMS)")
	print("                            Distributed under free software (GNU General Public Licence) ")
	print("                                   (c) 2015 GPL, Verson 15.1, All rights reserved.")
	print("*********************************************************************************************************************\n")
	# Defining different variables	
	global Ont, protgo, protargets
	Gappr = {'u':1, 'w':2, 'z':3, 'xr':4, 'xn':5, 'xl':6, 'r':7, 'n':8, 'l':9, 's':10, 'li':11}
	appnames = {'u': 'GO-universal', 'w': 'Wang et al.', 'z':'Zhang et al.', 'r': 'Resnik','xr':'XGraSM-Resnik', 'n': 'Nunivers', 'xn': 'XGraSM-Nunivers', 'l': 'Lin', 'xl': 'XGraSM-Lin','li': 'Li et al','s': 'Relevance'}
	
	print("Start processing your request on %s\n"%str(time.asctime(time.localtime())))
	
	tn = now = time.time()
	if len(args) < 2:
		print('Illegal number of arguments: at least two arguments required, no argument given.\nFor now, the process cannot be pursued: number of arguments error ...')
		sys.exit(1)
	elif  len(args)+len(kwargs) > 8:
		print('Illegal number of arguments: at most 8 arguments required, more than 8 arguments given.\nCheck different parameters or refer to the package documentation.\nFor now, the process cannot be pursued: number of arguments error ...')
		sys.exit(2)
	elif len(args)>=2:
		if not os.path.exists(os.path.join(os.path.dirname(__file__), args[0])) or not os.path.exists(os.path.join(os.path.dirname(__file__), args[1])):
			print("One of Target protein list or background file cannot be opened for reading or\nthere is argument value error. Check and try again ...")
			sys.exit(2)
		if len(args)>=3:
			kwargs['ontology'] = args[2]
			if len(args)>=4:
				kwargs['approach'] = args[3]
				if len(args)>=5:
					kwargs['score']= args[4]
					if len(args)>=6:
						kwargs['pvalue'] = args[5]
						if len(args)>=7:
							kwargs['drop'] = args[6]
							if len(args)>=8:
								kwargs['output'] = args[7]
		kwargs = _fixkwargs(kwargs, Gappr)

	SOntology = 1 + Ont.index(kwargs['ontology'])
	
	# Loading GO data
	getGOdata(SOntology)

	# Loading target and background protein sets
	try:
		readtargets(args[1])     # Reading protein targets dataset
		readbackground(args[0])  # Reading reference or background dataset
	except IOError:
		print("One of Target protein list or background file cannot be opened for reading or\nthere is argument value error. Check and try again ...")
		sys.exit(3)

	iset = set(protgo.keys())  # Meaning that Background or Reference protein set is not provided
	itarget = protargets & iset # Remembering to force ITarget to be included in the reference set, taking only those annotated

	N = len(iset); n = len(itarget)	
	try:
		tarterms = reduce(lambda x, y: x | y, [protgo[p] for p in itarget])
	except:
		tarterms = set()
	
	# Loading IC data
	if kwargs['approach'] in ['r','xr','n','xn','l','xl','s','li']:
		getICdata(SOntology, 1) # Loading GO data for Annotation based
	elif  kwargs['approach'] in ['u','w','z']:
		getICdata(SOntology, Gappr[kwargs['approach']]+1)  # For topology-based
	print("Setting parameters and loading data done, time elapsed:", int(round(time.time()-tn)), 'seconds...')

	print("\nComputing GO ID fuzzy frequency scores, this may take time...")
	tn = time.time()
	if 0.0 < kwargs['score'] < 1.0: # Fuzzy-cases
		Frequency = simProt(tarterms, itarget, iset, Gappr[kwargs['approach']], kwargs['score'])
	else:                           # Strict cases 0 and 1: traditional cases
		dispstr = ['Traditional::0 (considering true path rule)', 'Traditioanal::1 (effective occurence)']
		Frequency = simProt01(tarterms, itarget, iset, kwargs['score'])
		kwargs['approach'] = dispstr[int(kwargs['score'])]
	print("\rFrequency done, time elapsed:", int(round((time.time()-tn)/60)), 'minutes...'+85*' ')
	tn = time.time()
		
	# computing p-values using hypergeometric distribution and corrected p-values by Bonferroni
	print("\nComputing p-value now, this may take time...")
	Bonf = len(tarterms)
	P03 = {}; stringp = "Processing : %s %d%s"%(100*'.',0,'%'); count = 0; nt = len(Frequency)
	for t in Frequency:
		print(stringp, end=' ')
		count += 1
		if Frequency[t][0] > 0: Pv = 1-dst.hypergeom.cdf(Frequency[t][0]-1, N, Frequency[t][1], n)
		else: Pv = 1.0
		if Pv <= 0.0: Pv = 0.0 # The Pv can become negative because of floating point
		if Pv >= 1.0: Pv = 1.0 # The Pv can go greater than 1.0 because of floating point
		PvBf = Bonf*Pv 		   # Bonferroni multiple correction
		if PvBf >= 1.0: PvBf = 1.0
		P03[t] = (Pv, PvBf)
		perc = (count*100)//nt
		stringp = "\rProcessing : %s %d%s"%(perc*'='+(100-perc)*'.',perc,'%')
	print("\rProcessing : %s %d%s"%(100*'=',100,'%'), end=' ')
	print("\rComputing p-value done, time elapsed:", int(round(time.time()-tn)), 'seconds...'+75*' ')

	print("\nFiltering identified enriched GO annotations now, this may take time...")
	tn = time.time()
	# Filtering set of significant terms
	Fl03 = [(t,lcc[t][0]) for t in P03 if P03[t][1] < kwargs['pvalue']]
	Fl03 = [t[0] for t in sorted(Fl03, Sort)]
	Fl03 = filterSet(Fl03)
	
	# building output
	Results = []; S = set()
	for t in Fl03: # GO ID, GO term, Term Level, Frequency Background, Frequency Target, p-value, Bonferroni correction, 1/GO ID
		if not t in S:
			S.add(t)
			Results.append((ccind[t], termcc[t][0], lcc[t][0], Frequency[t][1], Frequency[t][0], P03[t][0], P03[t][1], '1'))
			try:
				for s in Fl03[t]:
					if not s in S:
						S.add(s)
						Results.append((ccind[s],termcc[s][0],lcc[s][0],Frequency[s][1],Frequency[s][0],P03[s][0],P03[s][1],ccind[t]))
			except:
				pass
	del S
	dtype = [('go1','S10'),('term','S500'),('level',int),('freqb',int),('freqt',int),('pvalue',float),('bonf',float),('out','S10')]
	Results = list(sort(array(Results,dtype=dtype), order= ['out','bonf','level']))
	print("Filtering identified enriched GO annotations and building output done, time elapsed:", int(time.time()-tn), 'seconds ...')
	
	print("\nIdentifying fuzzy enriched GO terms using :", "[gossfeat function from EnrichmentAnalysis.py module]")
	print("Total number of GO terms on the background:", len(reduce(lambda x, y: x | y, list(protgo.values()))))
	print("Total number of GO terms in the target set:", Bonf)
	print("Number of enriched GO terms detected      :", len(Results))
	print("Number of enriched no-redundant GO terms  :", len([a for a in Results if a[-1]=='1']))
	print("Semantic Similarity approaches used is    :", "[%s based approach]"%(appnames[kwargs['approach']] if kwargs['approach'] in appnames else kwargs['approach'],))

	# Returning a Python object: a list of tuples containing pairwise proteins and associated semantic similarity scores
	if kwargs['output']==2:
		print("\nIdentifying proteins related to particular GO annotations is  accomplished on %s"%str(time.asctime(time.localtime())))
		print("Total time elapsed is approximately:", (time.time()-now), 'seconds\n')
		print("\n************************************************************************************************************\n")
		return Results
	
	inputdata = inspect.getframeinfo(inspect.currentframe().f_back)[3][0].strip().split(',')[0].split('(')[1].strip()
	outputfile = inputdata.replace('\'','').replace('\"','').replace(':','_').split('.')[0].split('/')[-1].strip() + 'EA.txt'

	print("\nFinal results of fuzzy enrichment analysis can be found in the file: [%s]"%(outputfile,))
	try:
		fp = open(outputfile, 'w')
		fp.write("# Identifying fuzzy enriched GO terms using : [gossfeat function from EnrichmentAnalysis.py module]\n")
		fp.write("# Total number of GO terms on the background: %d\n"%(len(reduce(lambda x, y: x | y, list(protgo.values()))),))
		fp.write("# Total number of GO terms in the target set: %d\n"%(Bonf,))
		fp.write("# Number of enriched GO terms detected      : %d\n"%(len(Results),))
		fp.write("# Number of enriched no-redundant GO terms  : %d\n"%(len([a for a in Results if a[-1]=='1']),))
		fp.write("# Identifiying fuzzy enriched GO terms using: [%s] based approach\n"%(appnames[kwargs['approach']] if kwargs['approach'] in appnames else kwargs['approach'],))
		fp.write("# Legends: Ref_FF and Target_FF stand for number of occurrence on Reference and Target datasets, respectively\n") 
		outs = []; s = 45
		for out in Results:
			outs.append((out[0], out[1][:s], '%d'%out[2], '%d'%out[3], '%d'%out[4], '%1.5e'%out[5], '%1.5e'%out[6], out[7]))
			for i in range(1, int(ceil(len(out[1])*1.0/s))):
				try: outs.append(('',out[1][i*s:(i+1)*s], '', '', '', '', '', ''))
				except: pass
		headers = ['GO_ID', 'GO_Name', 'Level', 'Ref_FF', 'Target_FF', 'P-value', 'Correction', 'Decision']
		fp.write('%s'%tab(outs, headers, kwargs['tablefmt'], floatfmt="1.5e", stralign="left"))
		fp.close()
	except IOError:
		print("File cannot be opened in writing. Check possibly the writing permission and try again...")
		sys.exit(8)
	print("\nProcessing accomplished on %s"%str(time.asctime(time.localtime())))
	print("Total time elapsed is approximately", int(round((time.time()-now)/60)), 'minutes...') 
	print("\n*********************************************************************************************************************\n")

def _main(argv):
	"""
This constitutes a useful function for a user who chooses not to use 
the Python interpreter, but to run the package using a bash command
line. Note that this is only applicable in the case where user 
proteins' input data retrieved from files. In this case, identifying 
proteins related to GO IDs is achieved using the following command:

	python $(python -m site --user-site)/dagofun/EnrichmentAnalysis.py ReferenceFile TargetFile ontology approach score pvalue drop output

Different arguments are as explained in the termsim function (see pa-
ckage documentation). 
Moreover these arguments should be in order as shown in the command and 
as for commands under a Python interpreter, AnnotationData file con-
taining the user input list of proteins and their associated GO IDs and 
TargetIDs is another user input file containing GO IDs for which asso- 
proteins should be identified, and these two files must be provided. 
In case where other parameters are not provided, default parameters are 
used.

Assuming that the package was not installed, then:

$(python -m site --user-site)/dagofun/

should be replaced by the path to modules.
	"""
	if len(argv) <= 2:
		print('\nIllegal number of arguments: at least one argument required, no argument given.\nFor now, the process cannot be pursued: number of arguments error ...\n')
	elif  len(argv) > 9:
		print('\nIllegal number of arguments: at most 8 argument required, more than 4 arguments given.\nFor now, the process cannot be pursued: number of arguments error ...\n')
	else:
		if len(argv)==3: gossfeat(argv[1].strip(), argv[2].strip())
		elif len(argv)==4: gossfeat(argv[1].strip(), argv[2].strip(), ontology = argv[3].strip())
		else: # approach score mclust nclust drop output
			try:
				Approach = 'u'; Score = 0.3; Pvalue = 0.05; Drop = 0; Output = 1
				j = 4
				if len(argv) > j:
					Approach = argv[j].strip(); j += 1
					if len(argv) > j:
						try: 
							Score = float(argv[j].strip()); j += 1
							if not (0.0 < Score <= 1.0):
								print("\nThere is inconsistency. Please refer to the package documentation,\ncheck score argument provided, which should be a float in (0,1] interval ...\n")
								return
						except:
							print("\nThere is inconsistency. Please refer to the package documentation,\ncheck score argument provided, which should be a float in (0,1] interval ...\n")
							return
						if len(argv) > j:
							try: 
								Pvalue = float(argv[j].strip()); j += 1
								if not (0.0 < Pvalue <= 0.05):
									print("Check your level of significance cut-off, it must be a real number: 0 < pvalue <= 0.05.\n\nFor now, the process cannot be pursued: Cut-off pvalue error ...\n")
									return
							except:
								print("\nThere is inconsistency. Please refer to the package documentation,\ncheck pvalue argument provided, which should be a float in (0,1] interval ...\n")
								return
							if len(argv) > j:
								try:
									Drop = int(argv[j].strip()); j += 1
									if Drop != 0 and Drop != 1:
										print("Check the use of IEA evidence code variable <drop> which should be a Boolean:\n\t0 if all evidence code should be used and \n\t1 if IEA evidence code should be excluded.\n\nPlease check and try again ...")
										return
								except:
									print("\nThere is inconsistency. Please refer to the package documentation,\ncheck drop parameter provided ...\n")
									return
								if len(argv) > j:
									try:
										Output = int(argv[j].strip())
										if Output != 0:
											print("How do you want to output results should be an Enum 0 and 2:\n\t0 if results should be written in a file.\n\t2 for outputting a Python object for possible further usage.\n\nPlease check and try again ...")
											return
									except:
										print("\nThere is inconsistency. Please refer to the package documentation,\ncheck output parameter provided ...\n")
										return
				gossfeat(argv[1].strip(), argv[2].strip(), ontology = argv[3].strip(), approach = Approach, score = Score, pvalue = Pvalue, drop = Drop, output = Output)
			except:
				print("\nThere is inconsistency. Please refer to the package documentation,\ncheck the nappr argument provided and try again ...\n")

if __name__=='__main__':
	_main(sys.argv)

