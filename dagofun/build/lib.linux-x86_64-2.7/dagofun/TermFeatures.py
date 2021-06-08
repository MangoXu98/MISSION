#!/usr/bin/env python

"""
This python module is part of the DaGO-Fun tool, which is a tool for Gene 
Ontology-based functional analysis using term information content (IC)
measures.
This particular module allows users to retrieve features of a given list 
GO terms, including names, levels, status (active or obsolete), IC scores
of these GO terms provided the IC model. Four IC models are implemented
including: Annotation-based model and three topology-based model, namely
Wang et al., Zhang et al. and GO-unuversal models. Please, refer to the
PDF file for a complete description of these different IC models.

The main website for the G-DaGO-Fun package is 
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

__all__ = ['getTermFeatures']
__version__= "15.1"
__author__ = """Gaston K. Mazandu <gmazandu@{cbio.uct.ac.za, gmail.com}, kuzamunu@aims.ac.za>\n(c) 2015 All rights reserved."""

# Importing necessary python libraries
import sys, os, re, inspect, time
from math import ceil

# Importing package built-in modules
from .tabulate import tabulate as tab

# Importing external modules
try: # It is sufficient to import "cPickle" here once for reading or storing python binary files
	import pickle 
except ImportError:
    import pickle as cPickle

# Defining necessary global variables
goic_cc, goic_mf, goic_bp = {}, {}, {}
term_cc, term_mf, term_bp = {}, {}, {}
l_cc, l_mf, l_bp =  {}, {}, {}
i_cc, i_mf, i_bp = {}, {}, {}

Fam = ['AnnChar', 'Universal', 'WangIC', 'Zhang']

def readTermFeatures(sf, drop = 0):
	"""
		This is an engine loading all necessary GO parameters for use in retrieving GO term features.
	"""
	global l_cc, l_mf, l_bp, goic_cc, goic_mf, goic_bp, term_cc, term_mf, term_bp, i_cc, i_mf, i_bp		
	if sf==0 and drop==1:
		goic_cc = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOCC%sPartialI.ck'%(Fam[sf],)),'rb'))
		goic_mf = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOMF%sPartialI.ck'%(Fam[sf],)),'rb'))
		goic_bp = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOBP%sPartialI.ck'%(Fam[sf],)),'rb'))
	else:
		goic_cc = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOCC%sI.ck'%(Fam[sf],)),'rb'))
		goic_mf = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOMF%sI.ck'%(Fam[sf],)),'rb'))
		goic_bp = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOBP%sI.ck'%(Fam[sf],)),'rb'))

	l_cc = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOCCLevelAncestor.ck'),'rb'))
	l_mf = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOMFLevelAncestor.ck'),'rb'))
	l_bp = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOBPLevelAncestor.ck'),'rb'))

	term_cc = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOCCTerms.ck'),'rb'))
	term_mf = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOMFTerms.ck'),'rb'))
	term_bp = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOBPTerms.ck'),'rb'))

	i_cc = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOCCTermIndex.ck'),'rb'))
	i_mf = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOMFTermIndex.ck'),'rb'))
	i_bp = pickle.load(open(os.path.join(os.path.dirname(__file__),'data/GOBPTermIndex.ck'),'rb'))

	return

def _fixkwargs(dicts, allmod):
	"""
		This module checks different parameters *args and **kwargs and align them correctly in order
        to run termsim module.
	"""
	if len(dicts) > 3:
		print("Check out, 4 arguments are required but more than 4 arguments provided.")
		sys.exit(1)
			
	kwargs = {}
	if 'model' not in dicts: kwargs['model'] = 'u'
	elif type(dicts['model'])==str: 
		if dicts['model'].lower() in allmod: kwargs['model'] = dicts['model'].lower()
		else:
			print("Check notations of different IC model:\n\tu: For GO-universal\n\tw: For Wang et al.")
			print("\tz: For Zhang et al.\n\ta: For annotation-based")
			print("\nFor now, the process cannot be pursued: model key error ...")
			sys.exit(2)
	else:
		print("A model key should be a letter expressing a symbol of the IC model.\nRefer to the tool documentation for more information.\nFor now, the process cannot be pursued: model key error ...")
		sys.exit(3)

	if 'drop' not in dicts: kwargs['drop'] = 0 # In this case, this is relevant for annotation-based approaches
	elif dicts['drop'] is 0 or dicts['drop'] is 1: kwargs['drop'] = dicts['drop']
	else:
		print("Check the use of IEA evidence code variable <drop> which should be a Boolean:\n\t0 if all evidence code should be used and \n\t1 if IEA evidence code should be excluded.\n\nPlease check and try again ...")
		sys.exit(4)

	if 'output' not in dicts: kwargs['output'] = 1
	elif dicts['output'] is 0 or dicts['output'] is 1 or dicts['output'] is 2: kwargs['output'] = dicts['output']
	else:
		print("How do you want to output results is an Enum 0, 1, 2:\n\t1 if results should be displayed on the screen \n\t0 if results should be written in a file.\n\t2 for outputting a Python object for possible further usage.\n\nPlease check and try again ...")
		sys.exit(5)

	kwargs['tablefmt'] = 'rst' # One can also use 'grid' or other display formats!
	return kwargs

def readtermfile(File):
	"""
Reading the list of GO terms provided in a file 'File'
and returns a list of nice terms in the file
	"""
	term = re.compile("GO:\d{7}")
	fp = open(os.path.join(os.path.dirname(__file__),File),'r')
	niceterms = []; nonapp = set()
	for line in fp:
		ligne = line.strip()
		if not ligne: continue
		ligne = ligne.split(); ligne[0] = ligne[0].strip().upper()
		if ligne[0] in niceterms: continue
		if term.match(ligne[0]) and len(ligne[0])==10:
			niceterms.append(ligne[0])
		else: nonapp.add(ligne[0])
	fp.close()
	return niceterms, nonapp

def getTermFeatures(*args, **kwargs):
	"""
Description:
------------
This function retrieves Information Content (IC) scores and other GO
term features from the GO directed acyclic graph (DAG) structure.

Given a GO term or list/tuple or a file containing a list of GO IDs,
this function retrieves these characteristics of these GO terms in
the GO DAG, including their IC scores, provided the IC model. It uses 
GO-universal model ('u') by default, i.e., if no model is provided 
the GO-universal model is used.

*args* is a variable length argument, which can contain all -->
necessary parameters, in order: 
(1) a GO term, list/tuple or file containing a list of GO terms
(2) model, (3) drop and (4) output as described below.  

The *kwargs* can be used to set model, drop, outputs parameters.
This indicates that function argument (1) above is compulsory. 

IC model:
Symbol of four different IC models implemeted in this package are:
   'u': For the GO-Universal
   'w': For Wang et al.
   'z': For Zhang et al
   'a': For Annotation-based

* drop : boolean variable only useful in the context of Annotation-
based approach and it is set to 0 if Inferred from Electronic Anno-
tation (IEA) evidence code should be considered and to 1 otherwise.
 By default, it is set to 0.
     
*output: a boolean variable also set to 1 to display results on the 
screen and to 0 in a file. By default (output=1), results are disp-
layed on the screen, and finally default table display uses the pa- 
ckage module written by 'Sergey Astanin (s.astanin@gmail.com)' 
and collaborators. It used tablefmt="rst".
If results are written onto a file, the name of the file is basica-
lly the name of the first parameter in the function followed by TF
and  where ':' is replaced by '_', this is a case when this parame-
ter is a GO term.
Usage:
------
(a) getTermFeatures(InputData, model = 'u', drop = 0, output = 1)
	     
Examples:
--------
 (a) getTermFeatures('tests/TestTerms.txt')
 (b) getTermFeatures(['GO:0000001','GO:0048308', 'GO:0005385'], 'a')
	"""
	print("\n************************************************************************************************************")
	print("       Package G-DaGO-Fun: A General Gene Ontology Semantic Similarity based Functional Analysis Tool")
	print("           Computational Biology Group (CBIO) & African institute for Mathematical Sciences (AIMS)")
	print("                        Distribute under free software (GNU General Public Licence) ")
	print("                              (c) 2015 GPL, Verson 15.1 All rights reserved.")
	print("************************************************************************************************************\n")
	# Defining different variables
	global l_cc, l_mf, l_bp, goic_cc, goic_mf, goic_bp, term_cc, term_mf, term_bp, i_cc, i_mf, i_bp
	models = {'a':0, 'u':1, 'w':2, 'z':3}
	modelnames = {'a': 'Annotation-based', 'u': 'GO-universal', 'w': 'Wang et al.', 'z':'Zhang et al.'}
	sdpt = -1 # Indicates the source of datasets: list/tuple=1, file-string=0

	now = time.time()
	print("Searching for terms features starts on %s\n"%str(time.asctime(time.localtime())))
	# Checking different parameters 
	if len(args) < 1:
		print('Illegal number of arguments: at least one argument required, no argument given.\nFor now, the process cannot be pursued: number of arguments error ...')
		sys.exit(1)
	elif  len(args)+len(kwargs) > 4:
		print('Illegal number of arguments: at most 4 argument required, more than 4 arguments given.\nFor now, the process cannot be pursued: number of arguments error ...')
		sys.exit(2)
	elif 1<=len(args)<=4:
		if isinstance(args[0], (tuple, list)): sdpt = 1
		elif type(args[0])==str: sdpt = 0
		else:
			print("There is argument input value error. Check and try again ...")
			sys.exit(3)
		if len(args)>=2: 
			kwargs['model'] = args[1]
			if len(args)>=3: 
				kwargs['drop'] = args[2]
				if len(args)>=4: kwargs['output'] = args[3]		
		kwargs = _fixkwargs(kwargs, models)
	else:
		print("There is inconsistency. Please refer to the package documentation,\ncheck different parameters provided and try again ...")
		sys.exit(4)

	# Checking GO terms provided as input data 
	term = re.compile("GO:\d{7}")
	if sdpt==1:     # This means that we are dealing with a list of GO terms
		termlist = tuple([goid for goid in args[0] if term.match(goid)]) # Considering only those matching GO ID expression
		if not termlist: # if there is not GO ID then stop the process
			print("No GO ID identified in the set provided.\nFor now, the process cannot be pursued: Input value error ...")
			sys.exit(5)
	elif sdpt==0:	# This indicates that we are dealing with a file or only a single GO term
		try:
			termlist, nterms = readtermfile(args[0])
		except:
			if term.match(args[0]) and len(args[0])==10: termlist = (args[0],)
			else:
				print("No GO ID identified in the input provided.\nFor now, the process cannot be pursued: Input value error ...")
				sys.exit(6)
	else:
		print("There is inconsistency. Please refer to the package documentation,\ncheck different parameters provided and try again ...")
		sys.exit(7)

	# Now loading different GO term dataset
	readTermFeatures(models[kwargs['model']], kwargs['drop'])

	# Retrieving features of GO terms
	data = []; noident = set() 
	for p in termlist: # Retrieving GO ID features
		if p in i_cc:  # This means that the term is in cellular_component ontology
			gid = i_cc[p]
			if term_cc[gid][-1]: 
				try:
					data.append((p,'C',term_cc[gid][0],'A', '%d'%(l_cc[gid][0],),'%.2f'%round(goic_cc[gid],2)))
				except:
					data.append((p,'C',term_cc[gid][0],'A', '%d'%(l_cc[gid][0],), 'U'))
			else: data.append((p,'C',term_cc[gid][0],'O','U','U'))
		elif p in i_mf: # This means that the term is in molecular_function ontology
			gid = i_mf[p]
			if term_mf[gid][-1]: 
				try:
					data.append((p,'F',term_mf[gid][0],'A', '%d'%(l_mf[gid][0],), '%.2f'%round(goic_mf[gid],2)))
				except:
					data.append((p,'F',term_mf[gid][0],'A', '%d'%(l_mf[gid][0],), 'U'))
			else: data.append((p,'F',term_mf[gid][0],'O','U','U'))
		elif p in i_bp: # This means that the term is in biological_process ontology
			gid = i_bp[p]
			if term_bp[gid][-1]: 
				try:
					data.append((p,'P',term_bp[gid][0],'A', '%d'%(l_bp[gid][0],),'%.2f'%round(goic_bp[gid],2)))
				except: 
					data.append((p,'P',term_bp[gid][0],'A', '%d'%(l_bp[gid][0],), 'U'))
			else: data.append((p,'P',term_bp[gid][0],'O','U','U'))
		else:			#This means that the term is completely unknown! and is ignored from the list!
			noident.add(p)

	# Returning a Python object: a list of tuples containing terms' characteristics or features
	if kwargs['output']==2:
		print("\nSearching for terms' features accomplished on %s"%str(time.asctime(time.localtime())))
		print("Total time elapsed is approximately:", (time.time()-now), 'seconds\n')
		print("\n************************************************************************************************************\n")
		return data	
	# Outputting different results on the screen or in a file
	print("Retrieval of GO term characteristic using:", "[getTermFeatures function from the module TermFeatures.py]")
	print("Number of possible GO terms detected     :", len(data))
	print("IC scores are computed using             : [%s model]"%(modelnames[kwargs['model']],))
	if kwargs['output']:
		print("\nGO term features are displayed in the table below.\nIf possible, use full screen mode for more convenient visualization:")
	else:
		inputdata = inspect.getframeinfo(inspect.currentframe().f_back)[3][0].strip().split(',')[0].split('(')[1].strip() 
		outputfile = inputdata.replace('\'','').replace('\"','').replace(':','_').split('.')[0].split('/')[-1].strip() + 'TF.txt'
		print("GO term features can be found in the file: [%s]"%(outputfile,))
	
	outs = []; s = 50
	for out in data:
		outs.append((out[0], out[1], out[2][:s], out[3], out[4], out[5]))
		for i in range(1, int(ceil(len(out[2])*1.0/s))):
			try: outs.append(('', '', out[2][i*s:(i+1)*s], '', '', ''))
			except: pass
	headers = ['GO ID', 'Ontology', 'Name', 'Status', 'Level', 'IC Score']	
	if kwargs['output']: # Print on the screen
		print(tab(outs, headers, kwargs['tablefmt'], floatfmt=".2f", stralign="left"))
		print("\nLegend of possible letters as scores:")
		print("-------------------------------------")
		print("    F stands for molecular_function, P for biological_process and C for cellular_component.")
		print("    A indicate that the term is active and O the term is obsolete")
		print("    U only for IC score indicates that the GO ID was not used to annotated a protein.\n      This may occur for annotation-based model")
		print("    O indicates the GO ID is obsolete, in which the level is unknown: U")
		print("    U for level and IC score indicates the GO ID is completely unknown from the ontology")		
	else:
		try:
			fp = open(outputfile, 'w')
			fp.write("# Retrieval of GO term characteristic using: [getTermFeatures function from the module TermFeatures.py]\n")
			fp.write("# Number of possible GO terms detected     : %d\n"%(len(data),))
			fp.write("# IC scores are computed using             : %s model\n"%(modelnames[kwargs['model']],))
			fp.write("# F stands for molecular_function, P for biological_process and C for cellular_component.\n")
			fp.write("# A indicate that the term is active and O the term is obsolete\n")
			fp.write("# U only for IC score indicates that the GO ID was not used to annotated a protein. This may occur for annotation-based model.\n# U for level and IC score indicates the GO ID is completely unknown or obsolete from the ontology\n")
			fp.write( "# O indicates the GO ID is obsolete, in which the level is unknown: U\n\n")
			fp.write('%s'%tab(outs, headers, kwargs['tablefmt'], floatfmt=".2f", stralign="left"))
			fp.close()
		except IOError:
			print("File cannot be opened in writing. Check folder permission and try again ...")
			sys.exit(8)
	print("\nProcessing accomplished on %s"%str(time.asctime(time.localtime())))
	print("Total time elapsed is approximately:", (time.time()-now), 'seconds') 
	print("\n************************************************************************************************************\n")

def _main(argv):
	"""
This constitutes a useful function for a user who chooses not to use 
the Python interpreter, but to run the package using a bash command
line. Note that this is only applicable in the case where user terms' 
input data retrieved from the file. In this case, retrieving terms' 
features is achieved using the following command:

	python $(python -m site --user-site)/dagofun/TermFeatures.py InputData model drop output

Different argumentc are as explained in the termsim function (see pa-
ckage documentation). Note that arguments should be in order as shown 
in the command and as for commands under a Python interpreter, the 
InputData file containing the user input list of terms must be provi-
ded. In case where other parameters are not provided, default parame-
ters are used.

Assuming that the package was not installed, then:

$(python -m site --user-site)/dagofun/

should be replaced by the path to modules.
	"""
	if len(argv) <= 1:
		print('\nIllegal number of arguments: at least one argument required, no argument given.\nFor now, the process cannot be pursued: number of arguments error ...\n')
	elif  len(argv) > 5:
		print('\nIllegal number of arguments: at most 4 argument required, more than 4 arguments given.\nFor now, the process cannot be pursued: number of arguments error ...\n')
	else:
		if len(argv)==2: getTermFeatures(argv[1].strip())
		elif len(argv)==3: getTermFeatures(argv[1].strip(), ontology = argv[2].strip())
		elif len(argv)==4: 
			try: getTermFeatures(argv[1].strip(), model = argv[2].strip(), drop = int(argv[3].strip()))
			except:
				print("\nThere is inconsistency. Please refer to the package documentation,\ncheck 'drop' parameter provided and try again ...\n")
				return
		elif len(argv)==5:
			try:
				Drop = int(argv[3].strip()); Output = int(argv[4].strip())
				if Output != 0 and Output != 1:
					print("\nRunning this module using a command line is plausible only when\n the output parameter is set to 0 or 1. Change it and try again ...\n")
					return
				getTermFeatures(argv[1].strip(), model = argv[2].strip(), drop = Drop, output = Output)
			except:
				print("\nThere is inconsistency. Please refer to the package documentation,\ncheck drop and/or output parameters provided ...\n")

if __name__=='__main__':
	_main(sys.argv)
		

