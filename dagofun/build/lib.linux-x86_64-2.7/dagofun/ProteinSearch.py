#!/usr/bin/python

"""
This python file is part of the DaGO-Fun tool, which is a tool for Gene 
Ontology-based functional analysis using term information content 
measures.
This python code implements fuzzy protein search or identification based
on semantic similarity concepts. Furthemore, this is context independent 
and can be used for any GO annotated dataset as population background or
reference and is not a context-based search.

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
    (c) 2015 under free software (GPL), all rights reserved.
"""

__all__ = ["proteinfit"]
__version__= "15.1"
__author__ = """Gaston K. Mazandu (gmazandu@{cbio.uct.ac.za, gmail.com})"""

# Importing necessary python libraries
import sys, os, re, inspect, time

# Importing external python modules and package
from .tabulate import tabulate as tab
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
goic, lcc, gocf, termcc, icc, ccind, protgo = {}, {}, {}, {}, {}, {}, {}
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
	goic.clear(); gocf.clear()
		
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

def simProt(termset, app):
	""" Inputs: takes two parameters: the set of terms and Approach under consideration and returns ...
		Outputs: returns ...
	"""
	dtype = [('go1', int),('ss', float)]
	global protgo, lcc, goic, gocf
	getprot = {}
	if app==1: # GO-universal
		for t in termset:
			initset = set(lcc[t][1]+[t])
			getprot[t] = []
			for p in protgo:
				csim = []
				for s in protgo[p]:
					csimt = 0.0
					if s==t: csimt = 1.0
					else:
						A = initset.intersection(lcc[s][1]+[s])
						csimt = max([goic[a] for a in A])/max(goic[t], goic[s])
					csim.append((s,csimt))
				try: 
					csim = list(sort(array(csim,dtype=dtype), order= 'ss'))
					if csim[-1][1] > 0.0: getprot[t].append((p, csim[-1][1], mean([c[1] for c in csim]), csim[-1][0])) #Prot, high SS, AvgSS, termI
				except:
					pass
	elif app==2: # This is Wang et al. approach
		for t in termset:
			initset = set(gocf[t].keys()) #set(lcc[t][1]+[t])
			getprot[t] = []
			for p in protgo:
				csim = []
				for s in protgo[p]:
					csimt = 0.0
					if s==t: csimt = 1.0
					else:
						A = initset.intersection(list(gocf[s].keys())) %intersection(lcc[s][1]+[s])
						csimt = sum([gocf[t][a]+gocf[s][a] for a in A])/(goic[t]+goic[s])
					csim.append((s,csimt))
				try: 
					csim = list(sort(array(csim,dtype=dtype), order= 'ss'))
					if csim[-1][1] > 0.0: getprot[t].append((p, csim[-1][1], mean([c[1] for c in csim]), csim[-1][0])) #Prot, high SS, AvgSS, termI
				except:
					pass
	elif app==3: # This is Zhang et al. approach
		for t in termset:
			initset = set(lcc[t][1]+[t])
			getprot[t] = []
			for p in protgo:
				csim = []
				for s in protgo[p]:
					csimt = 0.0
					if s==t: csimt = 1.0
					else:
						A = initset.intersection(lcc[s][1]+[s])
						csimt = 2.0*max([goic[a] for a in A])/(goic[t] + goic[s])
					csim.append((s,csimt))
				try: 
					csim = list(sort(array(csim,dtype=dtype), order= 'ss'))
					if csim[-1][1] > 0.0: getprot[t].append((p, csim[-1][1], mean([c[1] for c in csim]), csim[-1][0])) #Prot, high SS, AvgSS, termI
				except:
					pass
	elif app==4: # XGraSM-Resnik
		MaxValue = max(goic.values())
		for t in termset:
			if not t in goic: continue
			initset = set(lcc[t][1]+[t])
			getprot[t] = []
			for p in protgo:
				csim = []
				for s in protgo[p]:
					csimt = 0.0
					if s==t: csimt = 1.0
					else:
						A = initset.intersection(lcc[s][1]+[s])
						A = A.difference([Head])
						if A: csimt = mean([goic[a] for a in A])/MaxValue
					csim.append((s,csimt))
				try: 
					csim = list(sort(array(csim,dtype=dtype), order= 'ss'))
					if csim[-1][1] > 0.0: getprot[t].append((p, csim[-1][1], mean([c[1] for c in csim]), csim[-1][0])) #Prot, high SS, AvgSS, termI
				except:
					pass
	elif app==5: # XGraSM-Nunivers
		for t in termset:
			if not t in goic: continue
			initset = set(lcc[t][1]+[t])
			getprot[t] = []
			for p in protgo:
				csim = []
				for s in protgo[p]:
					csimt = 0.0
					if s==t: csimt = 1.0
					else:
						A = initset.intersection(lcc[s][1]+[s])
						A = A.difference([Head])
						if A: csimt = mean([goic[a] for a in A])/max(goic[t], goic[s])
					csim.append((s,csimt))
				try: 
					csim = list(sort(array(csim,dtype=dtype), order= 'ss'))
					if csim[-1][1] > 0.0: getprot[t].append((p, csim[-1][1], mean([c[1] for c in csim]), csim[-1][0])) #Prot, high SS, AvgSS, termI
				except:
					pass
	elif app==6: # XGraSM-Lin
		for t in termset:
			if not t in goic: continue
			initset = set(lcc[t][1]+[t])
			getprot[t] = []
			for p in protgo:
				csim = []
				for s in protgo[p]:
					csimt = 0.0
					if s==t: csimt = 1.0
					else:
						A = initset.intersection(lcc[s][1]+[s])
						A = A.difference([Head])
						if A: csimt = 2*mean([goic[a] for a in A])/(goic[t]+goic[s])
					csim.append((s,csimt))
				try: 
					csim = list(sort(array(csim,dtype=dtype), order= 'ss'))
					if csim[-1][1] > 0.0: getprot[t].append((p, csim[-1][1], mean([c[1] for c in csim]), csim[-1][0])) #Prot, high SS, AvgSS, termI
				except:
					pass
	elif app==7: # Resnik
		MaxValue = max(goic.values())
		for t in termset:
			if not t in goic: continue
			initset = set(lcc[t][1]+[t])
			getprot[t] = []
			for p in protgo:
				csim = []
				for s in protgo[p]:
					csimt = 0.0
					if s==t: csimt = 1.0
					else:
						A = initset.intersection(lcc[s][1]+[s])
						csimt = max([goic[a] for a in A])/MaxValue
					csim.append((s,csimt))
				try: 
					csim = list(sort(array(csim,dtype=dtype), order= 'ss'))
					if csim[-1][1] > 0.0: getprot[t].append((p, csim[-1][1], mean([c[1] for c in csim]), csim[-1][0])) #Prot, high SS, AvgSS, termI
				except:
					pass
	elif app==8: # Nunivers
		for t in termset:
			if not t in goic: continue
			initset = set(lcc[t][1]+[t])
			getprot[t] = []
			for p in protgo:
				csim = []
				for s in protgo[p]:
					csimt = 0.0
					if s==t: csimt = 1.0
					else:
						A = initset.intersection(lcc[s][1]+[s])
						csimt = max([goic[a] for a in A])/max(goic[t], goic[s])
					csim.append((s,csimt))
				try: 
					csim = list(sort(array(csim,dtype=dtype), order= 'ss'))
					if csim[-1][1] > 0.0: getprot[t].append((p, csim[-1][1], mean([c[1] for c in csim]), csim[-1][0])) #Prot, high SS, AvgSS, termI
				except:
					pass
	elif app==9: # Lin
		for t in termset:
			if not t in goic: continue
			initset = set(lcc[t][1]+[t])
			getprot[t] = []
			for p in protgo:
				csim = []
				for s in protgo[p]:
					csimt = 0.0
					if s==t: csimt = 1.0
					else:
						A = initset.intersection(lcc[s][1]+[s])
						csimt = 2*max([goic[a] for a in A])/(goic[t]+goic[s])
					csim.append((s,csimt))
				try: 
					csim = list(sort(array(csim,dtype=dtype), order= 'ss'))
					if csim[-1][1] > 0.0: getprot[t].append((p, csim[-1][1], mean([c[1] for c in csim]), csim[-1][0])) #Prot, high SS, AvgSS, termI
				except:
					pass
	elif app==10: # Relevance
		for t in termset:
			if not t in goic: continue
			initset = set(lcc[t][1]+[t])
			getprot[t] = []
			for p in protgo:
				csim = []
				for s in protgo[p]:
					csimt = 0.0
					if s==t: csimt = 1.0
					else:
						A = initset.intersection(lcc[s][1]+[s])
						Mica = max([goic[a] for a in A])
						csimt = 2*Mica*(1.0-exp(-Mica))/(goic[t]+goic[s])
					csim.append((s,csimt))
				try: 
					csim = list(sort(array(csim,dtype=dtype), order= 'ss'))
					if csim[-1][1] > 0.0: getprot[t].append((p, csim[-1][1], mean([c[1] for c in csim]), csim[-1][0])) #Prot, high SS, AvgSS, termI
				except:
					pass
	elif app==11: # Li et al. (SimIC)
		for t in termset:
			if not t in goic: continue
			initset = set(lcc[t][1]+[t])
			getprot[t] = []
			for p in protgo:
				csim = []
				for s in protgo[p]:
					csimt = 0.0
					if s==t: csimt = 1.0
					else:
						A = initset.intersection(lcc[s][1]+[s])
						Mica = max([goic[a] for a in A])
						csimt = 2*Mica*(1.0-1.0/(1.0+Mica))/(goic[t]+goic[s])
					csim.append((s,csimt))
				try: 
					csim = list(sort(array(csim,dtype=dtype), order= 'ss'))
					if csim[-1][1] > 0.0: getprot[t].append((p, csim[-1][1], mean([c[1] for c in csim]), csim[-1][0])) #Prot, high SS, AvgSS, termI
				except:
					pass
	return getprot

def readannotationfile(FileName):
	""" 
		Reading protein file and constructing ProteinGO dictionary from the database, it takes 3 parameters and return 2 values:
		Inputs: FileName is the name of file containing protein pairs, Id is the protein Identifiers 1 for UniProt Ids and 2 for Genenames
				  and so indicating the ontology undeer consideration!
		Outputs: 
	"""
	global icc, termcc, Head, protgo
	protgo.clear()
	
	fp = open(os.path.join(os.path.dirname(__file__), FileName), 'r')
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

def readterms(FileName):
	""" 
		Reading the protein GO annotation file to protgo dictionary used to identify proteins fuzzy annotated to go terms
	"""
	termset = []
	fp = open(os.path.join(os.path.dirname(__file__), FileName), 'r')
	for line in fp:
		ligne = line.strip()
		if not ligne: continue
		ligne = ligne.split()
		term = ligne[0].strip()
		termset.append(term)
		del term
	fp.close()
	return termset

def _fixkwargs(dicts, Alapp):
	"""
		This module checks different parameters *args and **kwargs and align them correctly in order to run termsim module.
	"""
	if len(dicts) > 7:
		print("7 arguments are required but more than 7 arguments provided.\n") 
		sys.exit(1)
	
	kwargs = {}
	if 'ontology' not in dicts: kwargs['ontology'] = 'BP'
	elif dicts['ontology'].upper() in ['BP','MF','CC']: kwargs['ontology'] = dicts['ontology'].upper()
	else:
		print("Check your ontology, the TermSimilarity module uses the string:\n\t'BP': For Biological Process\n\t'MF': For Molecular Function\n\t'CC': For Cellular Component.\nunknown ontology was provided") 
		sys.exit(2)

	if 'approach' not in dicts: kwargs['approach'] = 'u'
	elif type(dicts['approach'])==str: 
		if dicts['approach'].lower() in Alapp: kwargs['approach'] = dicts['approach'].lower()
		else:
			print("Check notations of different approaches from the package documentation.\n\nFor now, the process cannot be pursued: approach key error ...\n") 
			sys.exit(3)
	else:
		print("Approach key should be a symbol of approach. Please check and try again ...\n") 
		sys.exit(4)

	if 'score' not in dicts: kwargs['score'] = 0.3
	elif isinstance(dicts['score'], (int, float)) and 0.0 <= dicts['score'] <= 1.0: kwargs['score'] = float(dicts['score'])
	else:
		print("Check your semantic similarity score cut-off, it must be a real number: 0 <= score <= 1.\n\nFor now, the process cannot be pursued: Cut-off score value error ...\n")
		sys.exit(5)

	if 'drop' not in dicts: kwargs['drop'] = 0
	elif dicts['drop'] is 0 or dicts['drop'] is 1: kwargs['drop'] = dicts['drop']
	else:
		print("Check the use of IEA evidence code variable <drop> which should be a Boolean:\n\t0 if all evidence code should be used and \n\t1 if IEA evidence code should be excluded.\n\nPlease check and try again ...\n") # raise ValueError
		sys.exit(6)
	if 'output' not in dicts: kwargs['output'] = 1
	elif dicts['output'] is 0 or dicts['output'] is 1 or dicts['output'] is 2: kwargs['output'] = dicts['output']
	else:
		print("How do you want to output results is an Enum 0, 1, 2:\n\t1 if results should be displayed on the screen \n\t0 if results should be written in a file.\n\t2 for outputting a Python object for possible further usage.\n\nPlease check and try again ...") # raise ValueError
		sys.exit(7)
	kwargs['tablefmt'] = 'rst' # One can also use 'grid' or other display formats!
	return kwargs


def _effpvalues(dicts, Tars, agree):
	global protgo
	n = len(protgo); P03 = {}; Bonf = len(Tars); NewKeep = {}
	Control = set() # Control to not do something several times
	if agree==0.0:
		for t in Tars:
			if t in Control: continue
			ss = 0.0; l = 0
			if t in dicts:
				l = len(dicts[t]); ss = mean([p[1] for p in dicts[t]])
				if l > 0: 
					pr = 1.0*len([p for p in protgo if t in protgo[p]])/n
					pvalue = 1-dst.binom.cdf(l-1, n, pr)
				else: pvalue = 1.0
			else: pvalue = 1.0
			if pvalue <= 0.0: pvalue = 0.0 # Controlling floatting point
			if pvalue >= 1.0: pvalue = 1.0
			PvBf = Bonf*pvalue # Bonferroni correction
			if PvBf >= 1.0: PvBf = 1.0
			P03[t] = (pvalue, PvBf, l, ss)
			Control.add(t)
			NewKeep[t] = sorted(dicts[t], Sort)		
	else:
		for t in Tars:
			if t in Control: continue
			l = 0; ss = 0.0
			if t in dicts:
				Temp = [p[1] for p in dicts[t] if p[1] >= agree]
				l = len(Temp); ss = mean(Temp) if Temp else 0.0
				if l > 0: 
					pr = 1.0*len([p for p in protgo if t in protgo[p]])/n
					pvalue = 1-dst.binom.cdf(l-1, n, pr)
				else: pvalue = 1.0
			else: pvalue = 1.0
			if pvalue <= 0.0: pvalue = 0.0
			if pvalue >= 1.0: pvalue = 1.0
			PvBf = Bonf*pvalue # Bonferroni correction
			if PvBf >= 1.0: PvBf = 1.0
			P03[t] = (pvalue, PvBf, l, ss)
			Control.add(t)
			NewKeep[t] = sorted([p for p in dicts[t] if p[1] >= agree], Sort)
	return P03, NewKeep

def proteinfit(*args, **kwargs):
	"""
This function retrieves genes or proteins contributing to a given processes
at a certain threshold or agreement level based of protein or gene annota-
tions.
*args* is a variable length argument, which is two strings representing the 
name of the file of background proteins, each with its GO ID annotations,
and the target GO ID file or list, or a dictionary with protein or gene IDs 
and keys and list ot tuples of GO IDs as values and the target GO ID file or
list.
 
*kwargs* can be used to set ontology, approach, drop, outputs and tablefmt 
arguments.

  >>> proteinfit(AnnotationData, TargetGOIDs, ontology='BP', approach='u', score=0.3, drop = 0)

The function outputs:
  (1) Summary statistics for different target GO IDs proteins, which includes 
following fields:
GO ID, GO term, Term Level, Number of proteins, Average SS, p-value and
Bonferroni correction displayed on the screen or in a file depending on the argument
output.
  (2) For each GO ID target, a summary statistics is provided a file named
using the GO ID under consideration with following fields:
Protein ID, GO-ID related to the term,  GO-IDs with high-SS, Maximum SS and
Average SS
  
 A default parameters are GO-universal ('u') for SS, score = 1 the threshold or 
agreement level, Considering all GO evidence codes (drop = 0) and display by 
default on the screen (outputs=1), and finally default table display 
(tablefmt="rst") see tabulate package written by 'Sergey Astanin' and collabo-
rators.
SS Approach
-----------
Symbols of different approach are:
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

* drop : boolean variable only useful in the context of Annotation based 
approach and it is set to 0 if Inferred from Electronic Annotation (IEA) 
evidence code should be considered and to 1 otherwise.  By default, it is 
set to 0.
     
*output: Results are displayed on the screen, and finally default table 
display uses the package module written by 'Sergey Astanin 
(s.astanin@gmail.com)' and collaborators. It used tablefmt="rst".
	
Example:
--------
	>>> targets = ['GO:0006355', 'GO:001905', 'GO:0001658']
	>>> proteinfit('tests/TestProteins.txt', target)
    >>> proteinfit('tests/TestProteins.txt', 'tests/TermSimTest.txt', 'BP', 'n')
	>>> background = {'Q5H9L2':['GO:0006355','GO:0006351'], 'P03891':['GO:0022904','GO:0044281','GO:0044237','GO:0006120'], 'Q5H9L2':['GO:0006355','GO:0006351']}
    >>> proteinfit(background, targets, approach='s', score = 0)
	>>> targets = ['GO:0009597', 'GO:0046718', 'GO:0032725', 'GO:0032727', 'GO:0047088', 'GO:0045416']
	>>> proteinfit('tests/SpecificRefSet1.txt', targets)
	>>> targets = ['GO:0006612', 'GO:0001666', 'GO:0009611', 'GO:0009409', 'GO:0002679', 'GO:0009626', 'GO:0050832', 'GO:0009410', 'GO:0016045']
	>>> proteinfit('tests/SpecificRefSet2.txt', targets, score=0.0)
	"""
	print("\n*****************************************************************************************************************")
	print("       Package G-DaGO-Fun: A General Gene Ontology Semantic Similarity based Functional Analysis Tool")
	print("           Computational Biology Group (CBIO) & African institute for Mathematical Sciences (AIMS)")
	print("                       Distributed under free software (GNU General Public Licence) ")
	print("                             (c) 2015 GPL, Verson 15.1, All rights reserved.")
	print("*****************************************************************************************************************\n")
	# Defining different variables	
	global Ont, protgo
	Gappr = {'u':1, 'w':2, 'z':3, 'xr':4, 'xn':5, 'xl':6, 'r':7, 'n':8, 'l':9, 's':10, 'li':11}
	appnames = {'u': 'GO-universal', 'w': 'Wang et al.', 'z':'Zhang et al.', 'r': 'Resnik','xr':'XGraSM-Resnik', 'n': 'Nunivers', 'xn': 'XGraSM-Nunivers', 'l': 'Lin', 'xl': 'XGraSM-Lin','li': 'Li et al','s': 'Relevance'}
	
	print("Start processing your request on %s\n"%str(time.asctime(time.localtime())))
	
	tn = now = time.time()
	sdpt = -1 # Indicates the source of datasets: dict-list/tuple=1, file-file=0, file-list/tuple=2, dict-file=3
	if len(args) < 2:
		print('Illegal number of arguments: at least two arguments required, %d argument provided.'%(len(args),))
		sys.exit(1)
	elif len(args)+len(kwargs) > 7:
		print('Illegal number of arguments: at most 7 arguments required, more than 7 arguments given.\nFor now, the process cannot be pursued: number of arguments error ...')
		sys.exit(1)
	elif 2<=len(args)<=6:
		if not os.path.exists(os.path.join(os.path.dirname(__file__), str(args[0]))) and type(args[0])!=dict:
			print ("Wrong format of Annotation data is detected or\nthere is argument value error. Check and try again ...")
			sys.exit(3)
		if not os.path.exists(os.path.join(os.path.dirname(__file__), str(args[1]))) and not isinstance(args[1],(list, tuple, str)):
			print ("Wrong format of Target GO IDs is detected or\nthere is argument value error. Check and try again ...")
			sys.exit(4)
		if len(args)>=3: 
			kwargs['ontology'] = args[2]
			if len(args)>=4: 
				kwargs['approach'] = args[3]
				if len(args)>=5: 
					kwargs['score'] = args[4]
					if len(args)>=6: kwargs['drop'] = args[5]						
		kwargs = _fixkwargs(kwargs, Gappr)
	else:
		print("There is inconsistency. Please refer to the package documentation.\n\nCheck different parameters provided and try again ...") 
		sys.exit(3)

	SOntology = 1 + Ont.index(kwargs['ontology'])
	
	# Loading GO data
	getGOdata(SOntology)

	# Loading Annotation data and GO ID targets
	try:
		readannotationfile(args[0])  # Reading reference or background dataset
	except:
		if type(args[0])==dict:
			for p in args[0]: 
				protgo[p] = set([icc[t] for t in args[0][p] if (t in icc) and termcc[icc[t]][-1] and icc[t]!=Head])
				if not protgo[p]: protgo.pop(p, None) # Delete safely the concept added when it does not have annotations
		else:
			print ("Check the file containing of Annotation data or the format of annotation data and try again ...")
			sys.exit(3)
	if not protgo:
		print("Possibly GO IDs in the annotation file are not found in the version of GO dataset used.\nThe process cannot be pursued ...")
		sys.exit(4)

	try:
		termtargets = readterms(args[1].strip())
	except:
		if type(args[1])==str:
			if len(args[1].strip())==10: termtargets = (args[1].strip(),)
			else:
				print("The target GO ID provided does not seem to be a GO ID.\nCheck it and try again...")
				sys.exit(5)
		elif isinstance(args[1], (tuple, list)): termtargets = tuple(args[1])
		else:
			print("Check the file containing of target GO IDs or the format of target GO IDs data and try again ...")
			sys.exit(6)
	TarTerms = list(); TarDict = {}; Nasty = set() # Nasty contains uncontrolled and like GO terms
	for t in termtargets:
		try:
			TarTerms.append(icc[t])
			TarDict[icc[t]] = t
		except:
			Nasty.add(t)
	if not TarTerms:
		print("Possibly GO IDs in the target set are not found in the version of GO dataset used.\nThe process cannot be pursued ...")
		sys.exit(7)
	
	# Loading IC data
	if kwargs['approach'] in ['r','xr','n','xn','l','xl','s','li']: # or Gappr[kwargs['approach']] > 3
		getICdata(SOntology, 1, kwargs['drop']) # Loading GO data for Annotation based
	elif  kwargs['approach'] in ['u','w','z']:
		getICdata(SOntology, Gappr[kwargs['approach']]+1)  # For topology-based
	
	KeepProteins = simProt(set(TarTerms), Gappr[kwargs['approach']])
	P03, New = _effpvalues(KeepProteins, set(TarTerms), kwargs['score'])
	del KeepProteins

	print("\nFuzzy Identification of proteins/genes using    :", "[proteinfit function from the module ProteinIdentification.py]")
	print("Total number of possible target GO IDs in the list:", len(TarTerms))
	print("Semantic Similarity approaches used is  :", "[%s] approach"%(appnames[kwargs['approach']],))

	inputdata = inspect.getframeinfo(inspect.currentframe().f_back)[3][0].strip().split(',')[0].split('(')[1].strip()
	outputfile = inputdata.replace('\'','').replace('\"','').replace(':','_').split('.')[0].split('/')[-1].strip() + 'PI.txt'

	Results = []
	for t in TarTerms: # GO ID, GO term, Term Level, Number of proteins, Average SS, p-value, Bonferroni correction
		#fp.write('%s\t%s\t%d\t%d\t%.5f\t%1.5e\t%1.5e\n'%(TarDict[t],termcc[t][0],lcc[t][0],P03[t][-2],P03[t][-1],P03[t][0],P03[t][1]))
		Results.append((TarDict[t], termcc[t][0], lcc[t][0], P03[t][-2], P03[t][-1], P03[t][0], P03[t][1]))
	# Returning a Python object: a list of tuples containing pairwise proteins and associated semantic similarity scores
	if kwargs['output']==2:
		print("\nIdentifying proteins related to particular GO annotations is  accomplished on %s"%str(time.asctime(time.localtime())))
		print("Total time elapsed is approximately:", (time.time()-now), 'seconds\n')
		print("\n************************************************************************************************************\n")
		return Results 
	
	outs = []; s = 35
	for out in Results:
		outs.append((out[0], out[1][:s], '%d'%out[2], '%d'%out[3], '%1.5f'%out[4], '%1.5e'%out[5], '%1.5e'%out[6]))
		for i in range(1, int(ceil(len(out[1])*1.0/s))):
			try: outs.append(('',out[1][i*s:(i+1)*s], '', '', '', '', '', ''))
			except: pass
	
	# Outputting proteins/genes associated to each GO ID in the potential list!
	for goindex in New: # Protein ACC, Gene Name, Description, GO-ID related to the term,  GO-IDs with high-SS, Maximum SS, Average SS
		fp = open(TarDict[goindex].replace(':','_')+'PI.txt', 'w')
		fp.write("Concept-ID\tGO-ID related to the term\tMaximum SS\tAverage SS\n")
		for p in New[goindex]:
		 	fp.write('%s\t%s\t%.5f\t%.5f\n'%(p[0], ccind[p[-1]], p[1], p[2]))
		fp.close()

	if kwargs['output']==1:
		headers = ['GO-ID', 'GO-term', 'Level', '# of Concepts', 'Avg SS', 'p-value', 'Cp-values']
		print(tab(outs, headers, kwargs['tablefmt'], floatfmt=".5f", stralign="left"))
		print("\nLegends:")
		print("----------------------------------------")
		print("Cp-values stand for corrected p-values done using Bonferroni multiple correction model.")
		print("This p-value is computed using the binomial distribution.")
	else:
		try:
			print("\nGenerale Statistics for each target GO ID can be found in the file: [%s]"%(outputfile,))
			fp = open(outputfile, 'w')
			fp.write("# \nFuzzy Identification of proteins/genes using    :", "[proteinfit function from the module ProteinIdentification.py\n")
			fp.write("# Total number of possible target GO IDs in the list: %d using the [%s] approach\n\n"%(len(TarTerms), appnames[kwargs['approach']]))
			fp.write("#Legends:\n")
			fp.write("----------------------------------------\n")
			fp.write("Cp-values stand for corrected p-values done using Bonferroni multiple correction model.\n")
			fp.write("This p-value is computed using the binomial distribution.\n")
			fp.write('%s'%tab(outs, headers, kwargs['tablefmt'], floatfmt=".5f", stralign="center"))
			fp.close()
		except IOError:
			print("File %s cannot be opened in writing. Check possibly the writing permission and try again ..."%(outputfile,))
			sys.exit(8)	
	print("\nProcessing accomplished on %s"%str(time.asctime(time.localtime())))
	print("Total time elapsed is approximately", (time.time()-now), 'seconds') 
	print("\n*****************************************************************************************************************\n")

def _main(argv):
	"""
This constitutes a useful function for a user who chooses not to use 
the Python interpreter, but to run the package using a bash command
line. Note that this is only applicable in the case where user 
proteins' input data retrieved from files. In this case, identifying 
proteins related to GO IDs is achieved using the following command:

	python $(python -m site --user-site)/dagofun/ProteinSearch.py AnnotationData TargetIDs ontology approach score drop output

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
	elif  len(argv) > 8:
		print('\nIllegal number of arguments: at most 8 argument required, more than 4 arguments given.\nFor now, the process cannot be pursued: number of arguments error ...\n')
	else:
		if len(argv)==3: proteinfit(argv[1].strip(), argv[2].strip())
		elif len(argv)==4: proteinfit(argv[1].strip(), argv[2].strip(), ontology = argv[3].strip())
		else: # approach score mclust nclust drop output
			try:
				Approach = 'u'; Score = 0.3; Drop = 0; Output = 1
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
									if Output != 0 and Output != 1:
										print("How do you want to output results should be an Enum 0, 1, 2:\n\t1 if results should be displayed on the screen \n\t0 if results should be written in a file.\n\t2 for outputting a Python object for possible further usage.\n\nPlease check and try again ...")
										return
								except:
									print("\nThere is inconsistency. Please refer to the package documentation,\ncheck output parameter provided ...\n")
									return
				proteinfit(argv[1].strip(), argv[2].strip(), ontology = argv[3].strip(), approach = Approach, score = Score, drop = Drop, output = Output)
			except:
				print("\nThere is inconsistency. Please refer to the package documentation,\ncheck the nappr argument provided and try again ...\n")

if __name__=='__main__':
	_main(sys.argv)

