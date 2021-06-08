#!/usr/bin/env python

"""
This python file is part of the DaGO-Fun tool, which is a tool for Gene 
Ontology-based functional analysis using term information content (IC) 
measures.
This particular python code implements all known Gene Ontology IC-based 
term semantic similarity measures and allows to use more than one 
approach. In fact, up to four (4) approaches can be simultaneously run.
Please, refer to the PDF file for a complete description of these 
different IC models.

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
    (c) 2015 under free software (GPL), All rights reserved.
"""

__all__ = ['termsim']
__version__= "15.1"
__author__ = """Gaston K. Mazandu (gmazandu@{cbio.uct.ac.za, gmail.com})\n(c) 2014 All rights reserved."""

# Importing necessary python libraries
import sys, os, re, inspect, time
from math import exp

# Importing package built-in modules
from .tabulate import tabulate as tab
from functools import reduce

# Importing external python modules
try: # It is sufficient to import "cPickle" here once for reading or storing python binary files
	import pickle 
except ImportError:
    import pickle as cPickle

# Defining necessary global variables
goic, lcc, gocf, termcc, icc = {}, {}, {}, {}, {}
Head = None
Fam = ['AnnChar', 'Universal', 'WangIC', 'Zhang']
Ont = ['BP', 'MF', 'CC']

def getICdata(so, sf, drop = 0):
	"""
		This is an engine loading all necessary GO parameters for use in the computation of similarity scores.
	"""
	global Head, lcc, goic, gocf, Fam, Ont, termcc, icc
	goic.clear(); lcc.clear(); gocf.clear(); termcc.clear(); icc.clear()
		
	if sf==1 and drop==1:
		goic = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%s%sPartialI.ck'%(Ont[so-1], Fam[sf-1])),'rb'))
	else:
		goic = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%s%sI.ck'%(Ont[so-1], Fam[sf-1])),'rb'))
	lcc =  pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sLevelAncestor.ck'%(Ont[so-1], )),'rb'))
	termcc = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sTerms.ck'%(Ont[so-1], )),'rb'))
	icc = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sTermIndex.ck'%(Ont[so-1], )),'rb'))
	if sf==3: 
		gocf = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GO%sWangCFI.ck'%(Ont[so-1],)),'rb'))
		
	if so==3: Head = icc['GO:0005575']
	elif so==2: Head = icc['GO:0003674']
	else: Head = icc['GO:0008150']	
	return

def retrieveTermSimilarity(pairgoids, app):
	"""
		This is the engine computing term semantic similarity value for a GO-ID pairs.
		The abs function is used just because sometime the floating zero comes with 
		negative sign!
	"""
	global Head, lcc, goic, gocf, termcc, icc
	data = {}
	if app==1: # GO-Universal
		for p in pairgoids:
			if p[0] in icc and p[1] in icc:
				id1 = icc[p[0]]; id2 = icc[p[1]]
				if termcc[id1][-1] and termcc[id2][-1]:
					if id1==id2: data[(p[0],p[1])] = 1.
					else:
						ancestt = set(lcc[id1][1]+[id1]).intersection(lcc[id2][1]+[id2])
						mica = max([goic[t] for t in ancestt])
						data[(p[0],p[1])] = mica/max(goic[id1], goic[id2])
				else: data[(p[0],p[1])] = 'O' # One of the terms is unknown in the current setting
			else: data[(p[0],p[1])] = 'U'     # This means that the term is completely unknown! and is ignored from the list!
	elif app==2: # Wang et al.
		for p in pairgoids:
			if p[0] in icc and p[1] in icc:
				id1 = icc[p[0]]; id2 = icc[p[1]]
				if termcc[id1][-1] and termcc[id2][-1]:
					ancestt = set(gocf[id1].keys()).intersection(list(gocf[id2].keys()))
					data[(p[0],p[1])] = abs(sum([gocf[id1][a]+gocf[id2][a] for a in ancestt])/(goic[id1]+goic[id2]))
				else: data[(p[0],p[1])] = 'O'
			else: data[(p[0],p[1])] = 'U'  #This means that the term is completely unknown! and is ignored from the list!
	elif app==3: # Zhang et al.
		for p in pairgoids:
			if p[0] in icc and p[1] in icc:
				id1 = icc[p[0]]; id2 = icc[p[1]]
				if termcc[id1][-1] and termcc[id2][-1]:
					if id1==id2: data[(p[0],p[1])] = 1.
					else:
						ancestt = set(lcc[id1][1]+[id1]).intersection(lcc[id2][1]+[id2])
						mica = max([goic[t] for t in ancestt])
						data[(p[0],p[1])] = abs(2*mica/(goic[id1]+goic[id2]))
				else: data[(p[0],p[1])] = 'O'
			else: data[(p[0],p[1])] = 'U' #This means that the term is completely unknown! and is ignored from the list!
	elif app==4: # XGraSM-Resnik
		MaxValue = max(goic.values())
		for p in pairgoids:
			if p[0] in icc and p[1] in icc:
				id1 = icc[p[0]]; id2 = icc[p[1]]
				if termcc[id1][-1] and termcc[id2][-1]:
					if id1 in goic and id2 in goic:
						if id1==id2: data[(p[0],p[1])] = 1.
						else:								
							ancestt = set(lcc[id1][1]+[id1]).intersection(lcc[id2][1]+[id2])
							ancestt = ancestt.difference([Head])
							if ancestt: mica = abs(reduce(lambda x, y: x+y, [goic[t]/MaxValue for t in ancestt]))/len(ancestt)
							else: mica = 0.0
							data[(p[0],p[1])] = mica
					else: data[(p[0],p[1])] = 'I'
				else: data[(p[0],p[1])] = 'O'
			else: data[(p[0],p[1])] = 'U' # This means that the term is completely unknown in the current setting
	elif app==5: # Nunivers-XGraSM
		for p in pairgoids:
			if p[0] in icc and p[1] in icc:
				id1 = icc[p[0]]; id2 = icc[p[1]]
				if termcc[id1][-1] and termcc[id2][-1]:
					if id1 in goic and id2 in goic:
						if id1==id2: data[(p[0],p[1])] = 1.
						else:
							ancestt = set(lcc[id1][1]+[id1]).intersection(lcc[id2][1]+[id2])
							ancestt = ancestt.difference([Head])
							if ancestt: mica = abs(reduce(lambda x, y: x+y, [goic[t] for t in ancestt]))/len(ancestt)
							else: mica = 0.0
							data[(p[0],p[1])] = mica/max(goic[id1],goic[id2])
					else: data[(p[0],p[1])] = 'I' # This means that the term was not used to annotated a proteins
				else: data[(p[0],p[1])] = 'O'     # This means these terms are obsolete
			else: data[(p[0],p[1])] = 'U'         #This means that the term is completely unknown! and is ignored from the list!
	elif app==6: # XGraSM-Lin
		for p in pairgoids:
			if p[0] in icc and p[1] in icc:
				id1 = icc[p[0]]; id2 = icc[p[1]]
				if termcc[id1][-1] and termcc[id2][-1]:
					if id1 in goic and id2 in goic:
						if id1==id2: data[(p[0],p[1])] = 1.
						else:
							ancestt = set(lcc[id1][1]+[id1]).intersection(lcc[id2][1]+[id2])
							ancestt = ancestt.difference([Head])
							if ancestt: mica = abs(reduce(lambda x, y: x+y, [goic[t] for t in ancestt]))/len(ancestt)
							else: mica = 0.0
							data[(p[0],p[1])] = 2*mica/(goic[id1]+goic[id2])
					else: data[(p[0],p[1])] = 'I' # This means that the term was not used to annotated a proteins
				else: data[(p[0],p[1])] = 'O'     # This means these terms are obsolete
			else: data[(p[0],p[1])] = 'U'         # This means that the term is completely unknown! and is ignored from the list!
	elif app==7: # Resnik
		MaxValue = max(goic.values())
		for p in pairgoids:
			if p[0] in icc and p[1] in icc:
				id1 = icc[p[0]]; id2 = icc[p[1]]
				if termcc[id1][-1] and termcc[id2][-1]:
					if id1 in goic and id2 in goic:
						if id1==id2: data[(p[0],p[1])] = goic[id1]/MaxValue
						else: 
							ancestt = set(lcc[id1][1]+[id1]).intersection(lcc[id2][1]+[id2])
							mica = abs(max([goic[t]/MaxValue for t in ancestt]))
							data[(p[0],p[1])] = mica
					else: data[(p[0],p[1])] = 'I'
				else: data[(p[0],p[1])] = 'O'
			else: data[(p[0],p[1])] = 'U'  #This means that the term is completely unknown! and is ignored from the list!
	elif app==8: # Nunivers
		for p in pairgoids:
			if p[0] in icc and p[1] in icc:
				id1 = icc[p[0]]; id2 = icc[p[1]]
				if termcc[id1][-1] and termcc[id2][-1]:
					if id1 in goic and id2 in goic: # Checking if terms have information content values for annotation-based!
						if id1==id2: data[(p[0],p[1])] = 1.
						else:
							ancestt = set(lcc[id1][1]+[id1]).intersection(lcc[id2][1]+[id2])
							mica = abs(max([goic[t] for t in ancestt]))
							data[(p[0],p[1])] = mica/max(goic[id1],goic[id2])
					else: data[(p[0],p[1])] = 'I'
				else: data[(p[0],p[1])] = 'O'
			else: data[(p[0],p[1])] = 'U' #This means that the term is completely unknown! and is ignored from the list!
	elif app==9: # Lin
		for p in pairgoids:
			if p[0] in icc and p[1] in icc:
				id1 = icc[p[0]]; id2 = icc[p[1]]
				if termcc[id1][-1] and termcc[id2][-1]:
					if id1 in goic and id2 in goic: # Checking if terms have information content values for annotation-based!
						if id1==id2: data[(p[0],p[1])] = 1.
						else:
							ancestt = set(lcc[id1][1]+[id1]).intersection(lcc[id2][1]+[id2])
							mica = abs(max([goic[t] for t in ancestt]))
							data[(p[0],p[1])] = 2*mica/(goic[id1]+goic[id2])
					else: data[(p[0],p[1])] = 'I'
				else: data[(p[0],p[1])] = 'O'
			else: data[(p[0],p[1])] = 'U' #This means that the term is completely unknown! and is ignored from the list!	
	elif app==10: # SimRel
		for p in pairgoids:
			if p[0] in icc and p[1] in icc:
				id1 = icc[p[0]]; id2 = icc[p[1]]
				if termcc[id1][-1] and termcc[id2][-1]:
					if id1 in goic and id2 in goic:
						if id1==id2: data[(p[0],p[1])] = 1.
						else:								
							ancestt = set(lcc[id1][1]+[id1]).intersection(lcc[id2][1]+[id2])
							mica = abs(max([goic[t] for t in ancestt]))
							data[(p[0],p[1])] = 2*mica*(1.0-exp(-mica))/(goic[id1]+goic[id2])
					else: data[(p[0],p[1])] = 'I'
				else: data[(p[0],p[1])] = 'O'
			else: data[(p[0],p[1])] = 'U' #This means that the term is completely unknown! and is ignored from the list!
	elif app==11: # Li et al.
		for p in pairgoids:
			if p[0] in icc and p[1] in icc:
				id1 = icc[p[0]]; id2 = icc[p[1]]
				if termcc[id1][-1] and termcc[id2][-1]:
					if id1 in goic and id2 in goic:
						if id1==id2: data[(p[0],p[1])] = 1.
						else:								
							ancestt = set(lcc[id1][1]+[id1]).intersection(lcc[id2][1]+[id2])
							mica = abs(max([goic[t] for t in ancestt]))
							data[(p[0],p[1])] = 2*mica*(1.0-1.0/(1.0+mica))/(goic[id1]+goic[id2])
					else: data[(p[0],p[1])] = 'I'
				else: data[(p[0],p[1])] = 'O'
			else: data[(p[0],p[1])] = 'U' #This means that the term is completely unknown! and is ignored from the list!
	return data

def readtermfile(FileName):
	""" 
		Reading the file containing GO ID pairs, it takes one parameter which is the file being read
        and return 2 objects:
		Inputs: FileName is the name of file containing GO ID pairs. These GO IDs must all belong to
                the same ontology (BP, MF or CC) and takes only up to 3000 Go ID pairs
		Outputs: set of GO ID pairs for which semantic similarity scores must be computed.
	"""
	term = re.compile("GO:\d{7}")
	fp = open(os.path.join(os.path.dirname(__file__),FileName),'r')
	nicepairs = []; nopairs = set()
	for line in fp:
		ligne = line.strip()
		if not ligne: continue
		ligne = ligne.split()
		if len(ligne)!=2: continue
		ligne[0] = ligne[0].strip().upper(); ligne[1] = ligne[1].strip().upper()
		if term.match(ligne[0]) and term.match(ligne[1]) and len(ligne[0])==10 and len(ligne[1])==10: 
			nicepairs.append((ligne[0],ligne[1]))
		else: nopairs.add((ligne[0],ligne[1]))
	fp.close()
	return nicepairs, nopairs

def _fixkwargs(dicts, allapp):
	"""
		This module checks different parameters *args and **kwargs and align them correctly in order
        to run termsim module.
	"""
	if len(dicts) > 4: 
		print("5 or 6 arguments are required but more than 6 arguments provided.") #raise IOError
		sys.exit(1)
	
	kwargs = {}
	if 'ontology' not in dicts: kwargs['ontology'] = 'BP'
	elif dicts['ontology'].upper() in ['BP','MF','CC']: kwargs['ontology'] = dicts['ontology'].upper()
	else:
		print("Check your ontology, the TermSimilarity module uses the string:\n\t'BP': For Biological Process\n\t'MF': For Molecular Function\n\t'CC': For Cellular Component.\nunknown ontology was provided") # raise ValueError
		sys.exit(2)

	if 'approach' not in dicts: kwargs['approach'] = ('u',)
	elif type(dicts['approach'])==str: 
		if dicts['approach'].lower() in allapp: kwargs['approach'] = (dicts['approach'].lower(),)
		else:
			print("Check notations of different approaches from the package documentation.\n\nFor now, the process cannot be pursued: approach key error ...") # raise ValueError
			sys.exit(3)
	elif isinstance(dicts['approach'], (tuple, list)):
		dicts['approach'] = [p.lower() for p in dicts['approach']]
		if set(dicts['approach']).issubset(allapp):
			kwargs['approach'] = (dicts['approach'][0],)
			for i in range(1,len(dicts['approach'])):
				if not dicts['approach'][i] in kwargs['approach'] and len(kwargs['approach']) < 4: 
					kwargs['approach'] += (dicts['approach'][i],)
		else:
			print("There might be a problem with approach symbols.\nCheck notations of different approaches from the package documentation.\n\nFor now, the process cannot be pursued: approach key error ...") # raise ValueError
			sys.exit(4)
	else:
		print("Approach key is either a symbol of approach, list or tuple of approach symbols.\n\nPlease check and try again ...") # raise TypeError
		sys.exit(5)

	if 'drop' not in dicts: kwargs['drop'] = 0 # In this case, this is relevant for annotation-based approaches
	elif dicts['drop'] is 0 or dicts['drop'] is 1: kwargs['drop'] = dicts['drop']
	else:
		print("Check the use of IEA evidence code variable <drop> which should be a Boolean:\n\t0 if all evidence code should be used and \n\t1 if IEA evidence code should be excluded.\n\nPlease check and try again ...") # raise ValueError
		sys.exit(6)
		
	if 'output' not in dicts: kwargs['output'] = 1
	elif dicts['output'] is 0 or dicts['output'] is 1 or dicts['output'] is 2: kwargs['output'] = dicts['output']
	else:
		print("How do you want to output results is an Enum 0, 1, 2:\n\t1 if results should be displayed on the screen \n\t0 if results should be written in a file.\n\t2 for outputting a Python object for possible further usage.\n\nPlease check and try again ...") # raise ValueError
		sys.exit(7)
	kwargs['tablefmt'] = 'rst' # One can also use 'grid' or other display formats!
	return kwargs

def termsim(*args, **kwargs):
	"""
This function retrieves semantic similarity scores between terms.
*args* is a variable length argument, which is a GO term ID pair --> 
*GO ID 1* and *GO ID2* or a string representing the name of the file 
containing GO term ID pairs for which semantic similarity (SS) scores 
must be retrieved. For example, each of the following is legal::

>>> termsim('GO:0000001','GO:0048308') # displays the SS value beween 
'GO:0000001'and 'GO:0048308' using a default ontology: biological 
process, a default SS approach: GO-universal, Considering all GO 
evidence codes (drop = 0) and display by default on the screen 
(output=1), and finally default table display (tablefmt="rst") see 
tabulate package written by 'Sergey Astanin (s.astanin@gmail.com)' 
and collaborators.

>>> termsim('path/to/file_of_GOID_pairs')

The *kwargs* can be used to set ontology, approach, ciea, outputs and 
tablefmt arguments. Here is an example::

>>> termsim('GO:0000001','GO:0048308', ontology='BP', approach=('u','r','w'))

SS Approach
-----------
Approach or list/tuple of approaches under consideration (up to four 
(4) approaches can be considered). If more than 4 are given, these
beyond the 4th valid symbols are ignored.
   'u': for the GO-Universal
   'w': for Wang et al.
   'z': for Zhang et al
   'r': for Resnik
   'xr': for XGraSM-Resnik
   'n': for Nunivers
   'xn': for XGraSM-Nunivers
   'l':  for Lin
   'xl': for XGraSM-Lin
   'li': for Lin enhancement by Li et al (SimIC).
   's': for Lin enhancement by Relevance (Schlicker et al.)

* drop : boolean variable only useful in the context of Annotation-
based approach and it is set to 1 if Inferred from Electronic Anno-
tation (IEA) evidence code should be removed and to 0 otherwise. By
default, it is set to 0.
     
*output: a boolean variable also set to 1 to display results on the 
screen and to 0 in a file. By default (output=1), results are disp-
layed on the screen, and finally default table display uses the pa- 
ckage module written by 'Sergey Astanin (s.astanin@gmail.com)' 
and collaborators. It used tablefmt="rst".
If results are written onto a file, the name of the file is basica-
lly the name of the first parameter in the function followed by SS
and  where ':' is replaced by '_', this is a case when this parame-
ter is a GO term.
	     
Examples:
--------
 >>> termsim('GO:0000001','GO:0048308', ontology = 'BP', approach = 'u', drop = 1, output=1)
 >>> termsim('GO:0000001','GO:0048308', BP, ('u', 'w', 'z', 'li'), 1, 1)
 >>> termsim(['GO:0000001','GO:0048308', 'GO:0048311', 'GO:0000002'], approach = ('u','z'))
 >>> termsim('tests/TermSimTest.txt', approach = ('u','z','li'))
	"""
	print("\n************************************************************************************************************")
	print("       Package G-DaGO-Fun: A General Gene Ontology Semantic Similarity based Functional Analysis Tool")
	print("           Computational Biology Group (CBIO) & African institute for Mathematical Sciences (AIMS)")
	print("                        Distribute under free software (GNU General Public Licence) ")
	print("                              (c) 2015 GPL, Verson 15.1, All rights reserved.")
	print("************************************************************************************************************\n")
	# Defining different variables
	global Ont
	Gappr = {'u':1, 'w':2, 'z':3, 'xr':4, 'xn':5, 'xl':6, 'r':7, 'n':8, 'l':9, 's':10, 'li':11}
	appnames = {'u': 'GO-universal', 'w': 'Wang et al.', 'z':'Zhang et al.', 'r': 'Resnik','xr':'XGraSM-Resnik', 'n': 'Nunivers', 'xn': 'XGraSM-Nunivers', 'l': 'Lin', 'xl': 'XGraSM-Lin','li': 'Li et al','s': 'Relevance'}
	term = re.compile("GO:\d{7}")

	now = time.time()
	print("Retrieving Term semantic similarity scores starts on %s\n"%str(time.asctime(time.localtime())))

	#print "    Checking different parameters ..."
	if len(args) < 1:
		print('Illegal number of arguments: at least one argument required, no argument given.\nFor now, the process cannot be pursued: number of arguments error ...')
		sys.exit(1)
	elif  len(args)+len(kwargs) > 6:
		print('Illegal number of arguments: at most 6 argument required, more than 6 arguments given.\nFor now, the process cannot be pursued: number of arguments error ...')
		sys.exit(1)
	elif len(args)==1:           # This means that the only argument is a file containing GO ID pairs
		if isinstance(args[0], (tuple, list)): # The argument provided is a list or tuple of GO IDs
			pairs = tuple([goid for goid in args[0] if term.match(goid)]) # Considering only those matching GO ID expression
			if not pairs:        # if there is not GO ID then stop the process
				print("No GO ID identified in the set provided.\nFor now, the process cannot be pursued: Input value error ...")
				sys.exit(1)
			elif len(pairs)==1:
				print("Only one possible GO ID identified in the set provided [%s]"%(pairs[0],))
				print("and the similarity between a term and itself should be 1.00000.")
				sys.exit(1)
			npairs = set(args[0])-set(pairs)
			pairs = [(pairs[i], pairs[j]) for i in range(len(pairs)) for j in range(i+1, len(pairs))]
			kwargs = _fixkwargs(kwargs, Gappr)
		elif type(args[0])==str: # This is then the only possibility is that the argument provided is a file of GO ID pairs
			try:
				pairs, npairs = readtermfile(args[0])
				kwargs = _fixkwargs(kwargs, Gappr)
			except IOError:
				print("The file %s cannot be opened for reading or\nthere is argument value error. Check and try again ..."%(args[0],))
				sys.exit(1)
		else:
			print("There is argument input value error. Check different parameters and try again, for now the process cannot be pursued ...")
			sys.exit(1)
	elif 2<=len(args)<=6: # This means that arguments are GO ID pairs or a file and ontology
		if isinstance(args[0], (tuple, list)):
			if isinstance(args[1], (tuple, list)): # The arguments are lists or tuple of GO IDs
				if len(args[0]) != len(args[1]):
					print("\nWarning:: Be aware of the fact that the two lists/tuples are not of the same length. \nSo, the Scores are computed between GO ID pairs from these lists or tuples, and the number of these pairs\ncorresponds to the length of the shortest list/tuple: %d pairs\n"%(len(args[0]) if len(args[0]) < len(args[1]) else len(args[1]),))
				pairs = list(zip(args[0], args[1]))
				pairs = tuple([goid for goid in pairs if term.match(goid[0]) and term.match(goid[1])]) # Matching GO ID expression
				if not pairs:        # if there is not GO ID then stop the process
					print("No GO ID pairs identified in the two sets provided when paired.\nFor now, the process cannot be pursued: Input value error ...")
					sys.exit(1)
				npairs = set([goid for goid in args[0] if not term.match(goid)])-set([goid for goid in args[1] if not term.match(goid)])
				if len(args)>=3: 
					kwargs['ontology'] = args[2]
					if len(args)>=4: 
						kwargs['approach'] = args[3]
						if len(args)>=5: 
							kwargs['drop'] = args[4]
							if len(args)>=6: kwargs['output'] = args[5]
				kwargs = _fixkwargs(kwargs, Gappr)
			elif type(args[1])==str:  # This means that this should be the ontology
				if len(args)==6:
					print("There is argument input value error: Error of number of parameters\n. First Check different parameters, but now the process cannot be pursued...")
					sys.exit(1)
				pairs = tuple([goid for goid in args[0] if term.match(goid)]) # Considering only those matching GO ID expression
				if not pairs:        # if there is not GO ID then stop the process
					print("No GO ID identified in the set provided.\nFor now, the process cannot be pursued: Input value error ...")
					sys.exit(1)
				elif len(pairs)==1:
					print("Only one possible GO ID identified in the set provided [%s]"%(pairs[0],))
					print("and the similarity between a term and itself should be 1.00000.")
					sys.exit(1)
				npairs = set(args[0])-set(pairs)
				pairs = [(pairs[i], pairs[j]) for i in range(len(pairs)) for j in range(i+1, len(pairs))]
				kwargs['ontology'] = args[1]
				if len(args)>=3: 
					kwargs['approach'] = args[2]
					if len(args)>=4: 
						kwargs['drop'] = args[3]
						if len(args)==5: kwargs['output'] = args[4]
				kwargs = _fixkwargs(kwargs, Gappr)
			else:
				print("There is argument input value error. Check different parameters and try again.\nFor now the process cannot be pursued ...")
				sys.exit(1)			
		elif type(args[0])==str and type(args[1])==str: # Either they are all GO terms or a file and ontology
			if term.match(args[0]) and term.match(args[1]) and len(args[0])==10 and len(args[1])==10:
				pairs, npairs = [args[:2]], []
				if len(args)>=3: 
					kwargs['ontology'] = args[2]
					if len(args)>=4: 
						kwargs['approach'] = args[3]
						if len(args)>=5: 
							kwargs['drop'] = args[4]
							if len(args)==6: kwargs['output'] = args[5]
				kwargs = _fixkwargs(kwargs, Gappr)
			else:
				if len(args)==6:
					print("There is argument input value error: Error of number of parameters\n. First Check different parameters, but now the process cannot be pursued...")
					sys.exit(1)
				try:
					pairs, npairs = readtermfile(args[0])
					kwargs['ontology'] = args[1]
					if len(args)>=3: 
						kwargs['approach'] = args[2]
						if len(args)>=4: 
							kwargs['drop'] = args[3]
							if len(args)==5: kwargs['output'] = args[4]
					kwargs = _fixkwargs(kwargs, Gappr)
				except IOError:
					print("The file %s cannot be opened for reading or\nthere is argument value error. Check and try again ..."%(args[0],))
					sys.exit(1)
		else:
			print("There is argument input value error. Check different parameters and try again.\nFor now the process cannot be pursued...")
			sys.exit(1)
	else:
		print("There is inconsistency. Please refer to the package documentation.\n\nCheck different parameters provided and try again ...") 
		sys.exit(1)
			
	sontology = 1 + Ont.index(kwargs['ontology'])	
	sfam = {1:set([a for a in kwargs['approach'] if a in ['r','xr','n','xn','l','xl','li','s']]), 2:set([a for a in kwargs['approach'] if a in ['u','w','z']])}

	#print "    Now loading different GO term dataset ..."
	data = {}
	if sfam[1]:
		getICdata(sontology, 1, kwargs['drop']) # Loading GO data for Annotation based
		for sapproach in sfam[1]:
			data[sapproach] = retrieveTermSimilarity(pairs, Gappr[sapproach])
	if sfam[2]:
		for sapproach in sfam[2]:
			getICdata(sontology, Gappr[sapproach]+1)  # For topology-based
			data[sapproach] = retrieveTermSimilarity(pairs, Gappr[sapproach])

	#print "    Computing Semantic Similarity Scores of GO term pairs ...\n"
	outs = []
	for p in pairs:
		outs.append([p[0],p[1]])
		for a in kwargs['approach']:
			outs[-1].append(data[a][p])

	# Returning a Python object: a list of tuples containing pairwise terms and associated semantic similarity scores
	if kwargs['output']==2:
		print("\nRetrieving term pairwise semantic similarity scores accomplished on %s"%str(time.asctime(time.localtime())))
		print("Total time elapsed is approximately:", (time.time()-now), 'seconds\n')
		print("\n************************************************************************************************************\n")
		return outs 
	# Outputting different results
	print("Computing GO term similarity scores using:", "[termsim function from the module TermSimilarity.py]")
	print("Number of possible GO term pairs detected:", len(outs))
	print("Different Semantic Similarity approaches :", ", ".join([appnames[a] for a in kwargs['approach']]))
	if kwargs['output']:
		print("\nGO term features are displayed in the table below.\nIf possible, use full screen mode for more convenient visualization:")
	else:
		inputdata = inspect.getframeinfo(inspect.currentframe().f_back)[3][0].strip().split(',')[0].split('(')[1].strip()
		outputfile = inputdata.split('/')[-1].split('.')[0].replace('\'','').replace('\"','').replace(':','_') + 'SS.txt'
		print("Term SS scores can be found in the file  : [%s]"%(outputfile,))

	headers = ['GO-ID1', 'GO-ID2']+[appnames[a] for a in kwargs['approach']]	
	if kwargs['output']: # Print on the screen
		print(tab(outs, headers, kwargs['tablefmt'], floatfmt=".5f", stralign="center"))
		print("\nLegend of possible letters as scores:")
		print("----------------------------------------")
		print("    I: indicates that at least one of the GO IDs was not used to annotated a protein.\n\t   This may occur for annotation-based approaches")
		print("    O: indicates that at least one of the GO ID is obsolete")
		print("    U: indicates that at least one of the GO IDs is completely unknown from the selected sub-ontology")		
	else:
		try:
			fp = open(outputfile, 'w')
			fp.write("# Computing term semantic similarity for following approach(es):\n")
			for a in kwargs['approach']:
				fp.write("# %s: %s\n"%(a, appnames[a]))
			fp.write("# Here is the legend of possible letters coming as scores:\n")
			fp.write("#\tI: indicates that at least one of the GO IDs was not used to annotated a protein.\n\t   This may occur for annotation-based approaches\n")
			fp.write("#\tO: indicates that at least one of the GO ID is obsolete\n")
			fp.write("#\tU: indicates that at least one of the GO IDs is completely unknown from the selected sub-ontology\n\n")
			fp.write('%s'%tab(outs, headers, kwargs['tablefmt'], floatfmt=".5f", stralign="center"))
			fp.close()
		except IOError:
			print("File cannot be opened in writing. Check possibly the writing permission and try again ...")
			sys.exit(8)
	print("\nProcessing accomplished on %s"%str(time.asctime(time.localtime())))
	print("Total time elapsed is:", (time.time()-now), 'seconds') 
	print("\n************************************************************************************************************\n")

def _main(argv):
	"""
This constitutes a useful function for a user who chooses not to use 
the Python interpreter, but to run the package using a bash command
line. Note that this is only applicable in the case where user terms' 
input data retrieved from the file. In this case, retrieving terms' 
features is achieved using the following command:

	python $(python -m site --user-site)/dagofun/TermSimilarity.py FileName ontology nappr approach drop output

Different argumentc are as explained in the termsim function (see pa-
ckage documentation). Except that nappr which is the new argument and
represents the number of approaches to be executed as the module can 
retrieve term semantic similarity scores using more than one approach 
and can go up to four different approaches. These different approaches 
are then provided just after providing this number.
Moreover that arguments should be in order as shown in the command and 
as for commands under a Python interpreter, the FileName file contai-
ning the user input list of term pairs must be provided. In case where 
other parameters are not provided, default parameters are used.

Assuming that the package was not installed, then:

$(python -m site --user-site)/dagofun/

should be replaced by the path to modules.
	"""
	if len(argv) <= 1:
		print('\nIllegal number of arguments: at least one argument required, no argument given.\nFor now, the process cannot be pursued: number of arguments error ...\n')
	elif  len(argv) > 10:
		print('\nIllegal number of arguments: at most 9 argument required, more than 4 arguments given.\nFor now, the process cannot be pursued: number of arguments error ...\n')
	else:
		if len(argv)==2: termsim(argv[1].strip())
		elif len(argv)==3: termsim(argv[1].strip(), ontology = argv[2].strip())
		else: 
			try:
				nappr = int(argv[3])
				appr = []; j = 4
				for i in range(nappr):
					try:
						tapp = argv[4+i].strip(); j += 1
						if not tapp in appr: appr.append(tapp)
					except:
						print("\nThere is inconsistency between the number of approaches provided and expected \nnumber of approaches. Please refer to the package documentation,\ncheck nappr and approaches provided, and try again ...\n")
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
					print ("\nRunning this module using a command line is plausible only when\n the output parameter is set to 0 or 1. Change it and try again ...\n")
					return
				if nappr != len(appr):
					print("\nWarning: Please note that the number of approaches provided and the actual number of approaches are different.\nThe list of approaches provided is redundant.\n")
				termsim(argv[1].strip(), ontology = argv[2].strip(), approach = tuple(appr), drop = Drop, output = Output)
			except:
				print("\nThere is inconsistency. Please refer to the package documentation,\ncheck the nappr argument provided and try again ...\n")
	
if __name__=='__main__':
	_main(sys.argv)
	
