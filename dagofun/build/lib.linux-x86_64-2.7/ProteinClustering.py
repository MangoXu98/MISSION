#!/usr/bin/env python
"""
This python module is part of the DaGO-Fun tool, which is a tool for Gene 
Ontology-based functional analysis using term information content 
measures.
This python code implements hierarchical, Graph spectral (kmeans) and 
model or communuity-based clustering of genes and proteins based on their 
Gene Ontology annotations. Hierarchical and Graph spectral approaches are 
used as implemented under scipy and community-based approach is implemen-
ted using following modules as written by 
Thomas Aynaud <thomas.aynaud@lip6.fr>, 2009: "partition_at_level", 
"best_partition", "generate_dendogram", "induced_graph" .

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

__all__ = ["proteinfct"]
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

try: # It is sufficient to import "scipy" here once for some useful mathematical objects, functions and algorithm needed.
	from numpy import exp, array, zeros, diag, unique, mean, sqrt, dot, ceil
	from scipy.cluster import hierarchy
	from scipy.spatial import distance
	from scipy.cluster.vq import kmeans2 
	from scipy.linalg import svd
except ImportError:
	raise ImportError("The library scipy is required for some numerical computations. \nPlease, install scipy library and try again ...")

try: # It is sufficient to import "networkx" here once for some useful graph or network application
	import networkx as nx
except ImportError:
	raise ImportError("The library networkx is required for some graph or network objects. \nPlease, install it and try again ...")

try: # It is sufficient to import "matplotlib" here once for plotting.
	import matplotlib.pylab as plt 
except ImportError:
	raise ImportError("The library matplotlib is required for plotting. \nPlease, install it and try again ...")

# Defining necessary global variables
__PASS_MAX = -1
__MIN = 0.0000001

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
		tt = [(a, b) for a in protgo[p[0]] for b in protgo[p[1]]]
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
		tt = [(a, b) for a in protgo[p[0]] for b in protgo[p[1]]]
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
		tt = [(a, b) for a in protgo[p[0]] for b in protgo[p[1]]]
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
		tt = [(a, b) for a in protgo[p[0]] for b in protgo[p[1]]]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				mica = max([goic[t] for t in ancestt])
				TermSim[(a, b)]= TermSim[(b, a)]  = mica/MaxValue
				del ancestt, mica
	elif app == 8: # Nunivers
		TermSim.clear()
		tt = [(a, b) for a in protgo[p[0]] for b in protgo[p[1]]]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				mica = max([goic[t] for t in ancestt])
				TermSim[(a,b)] = TermSim[(b,a)] = mica/max(goic[a], goic[b])
				del ancestt, mica
	elif app == 9: # Lin
		TermSim.clear()
		tt = [(a, b) for a in protgo[p[0]] for b in protgo[p[1]]]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				mica = max([goic[t] for t in ancestt])
				TermSim[(a,b)] = TermSim[(b,a)] = 2*mica/(goic[a]+goic[b])
				del ancestt, mica
	elif app == 10: # SimRel
		TermSim.clear()
		tt = [(a, b) for a in protgo[p[0]] for b in protgo[p[1]]]
		for (a, b) in tt:
			if a == b: TermSim[(a,b)] = 1.0
			else:
				ancestt = set(lcc[a][1]+[a]).intersection(lcc[b][1]+[b])
				mica = max([goic[t] for t in ancestt])
				TermSim[(a, b)] = TermSim[(b,a)] = 2*mica*(1.0-exp(-mica))/(goic[a]+goic[b])
				del ancestt, mica
	elif app == 11: # Li et al.
		TermSim.clear()
		tt = [(a, b) for a in protgo[p[0]] for b in protgo[p[1]]]
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
			p0terms = p0terms.difference([Head]); p1terms = p1terms.difference([Head]);
			sanc = sum([goic[t] for t in p0terms.intersection(p1terms)])/sum([goic[t] for t in p0terms.union(p1terms)])
			data = round(sanc,5)	
			del p0terms, p1terms, sanc
		except:
			pass
	elif sem==2: # SimDIC
		try:
			p0terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[0]]])
			p1terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[1]]])
			p0terms = p0terms.difference([Head]); p1terms = p1terms.difference([Head]);
			sanc1 = sum([goic[t] for t in p0terms.intersection(p1terms)]); sanc2 = sum([goic[t] for t in p0terms.union(p1terms)])
			data = round(2.0*sanc1/(sanc2 + sanc1),5)
			del p0terms, p1terms, sanc1, sanc2
		except:
			pass
	elif sem==3: # SimUIC
		try:
			p0terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[0]]])
			p1terms = reduce(lambda x, y: x.union(y), [set(lcc[t][1]+[t]) for t in protgo[p[1]]])
			p0terms = p0terms.difference([Head]); p1terms = p1terms.difference([Head]);
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
			tunion = list(p0terms | p1terms)
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
			tunion = list(p0terms | p1terms)
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
			au = [TermSim[(s,t)] for t in protgo[p[0]] for s in protgo[p[1]]]
			data = round(sum(au)/len(au),5)
			del au
		except:
			pass
	elif sem==11: # Best Match Average
		try:
			au1 = [max([TermSim[(s,t)] for t in protgo[p[1]]]) for s in protgo[p[0]]]
			au2 = [max([TermSim[(s,t)]for t in protgo[p[0]]]) for s in protgo[p[1]]]
			data = round((mean(au1)+mean(au2))/2.0,5)
			del au1, au2
		except:
			pass
	elif sem==12: # Average Best Matches
		try:
			au1 = [max([TermSim[(s,t)] for t in protgo[p[1]]]) for s in protgo[p[0]]]
			au2 = [max([TermSim[(s,t)] for t in protgo[p[0]]]) for s in protgo[p[1]]]
			au = au1+au2
			data = round(mean(au),5)
			del au1, au2, au
		except:
			pass
	elif sem==13: # Best Match Maximum (RCMax) or MHDF
		try:
			au1 = [max([TermSim[(s,t)] for t in protgo[p[1]]]) for s in protgo[p[0]]]
			au2 = [max([TermSim[(s,t)] for t in protgo[p[0]]]) for s in protgo[p[1]]]
			data = round(max(mean(au1), mean(au2)),5)
			del au1, au2
		except:
			pass
	elif sem==14: # HDF
		try:
			au1 = [max([TermSim[(s,t)] for t in protgo[p[1]]]) for s in protgo[p[0]]]
			au2 = [max([TermSim[(s,t)] for t in protgo[p[0]]]) for s in protgo[p[1]]]
			data = round(1.0-max(1.0-min(au1), 1.0-min(au2)),5)
			del au1, au2
		except:
			pass
	elif sem==15: # VHDF
		try:
			au1 = [(1.0-max([TermSim[(s,t)] for t in protgo[p[1]]]))**2 for s in protgo[p[0]]]
			au2 = [(1.0-max([TermSim[(s,t)] for t in protgo[p[0]]]))**2 for s in protgo[p[1]]]
			data = round(1.0-(sqrt(sum(au1)/len(au1)) + sqrt(sum(au1)/len(au1)))/2.0, 5)
			del au1, au2
		except:
			pass
	elif sem==16: # Maximum
		try:
			au = [TermSim[(s,t)] for t in protgo[p[0]] for s in protgo[p[1]]]
			data = round(max(au),5)
			del au
		except:
			pass
	return data


class Status :
	""" Handling several data in one structure """
	node2com = {}
	total_weight = 0
	internals = {}
	degrees = {}
	gdegrees = {}
		
	def __init__(self):
		self.node2com = dict([])
		self.total_weight = 0
		self.degrees = dict([])
		self.gdegrees = dict([])
		self.internals = dict([])
		self.loops = dict([])
	
	def copy(self) :
		"""Perform a deep copy of status"""
		new_status = Status()
		new_status.node2com = self.node2com.copy()
		new_status.internals = self.internals.copy()
		new_status.degrees = self.degrees.copy()
		new_status.gdegrees = self.gdegrees.copy()
		new_status.total_weight = self.total_weight

	def init(self, graph, part = None) :
		"""Initialize the status of a graph with every node in one community"""
		count = 0
		self.node2com = dict([])
		self.total_weight = 0
		self.degrees = dict([])
		self.gdegrees = dict([])
		self.internals = dict([])
		self.total_weight = graph.size(weight = 'weight')
		if part == None :
			for node in graph.nodes() :
				self.node2com[node] = count
				deg = float(graph.degree(node, weight = 'weight'))
				self.degrees[count] = deg
				self.gdegrees[node] = deg
				self.loops[node] = float(graph.get_edge_data(node, node, {"weight":0}).get("weight", 1))
				self.internals[count] = self.loops[node]
				count = count + 1
		else :
			for node in graph.nodes() :
				com = part[node]
				self.node2com[node] = com
				deg = float(graph.degree(node, weigh = 'weight'))
				self.degrees[com] = self.degrees.get(com, 0) + deg
				self.gdegrees[node] = deg
				inc = 0.
				for neighbor, datas in graph[node].items() :
					weight = datas.get("weight", 1)
					if part[neighbor] == com :
						if neighbor == node : inc += float(weight)
						else : inc += float(weight) / 2.
				self.internals[com] = self.internals.get(com, 0) + inc

def __modularity(status) :
	""" Compute the modularity of the partition of the graph faslty using status precomputed """
	links = float(status.total_weight)
	result = 0.
	for community in set(status.node2com.values()) :
		in_degree = status.internals.get(community, 0.)
		degree = status.degrees.get(community, 0.)
		if links > 0 : result += in_degree / links - ((degree / (2.*links))**2)
	return result

def __renumber(dictionary) :
	""" Renumber the values of the dictionary from 0 to n """
	count = 0
	ret = dictionary.copy()
	new_values = dict([])

	for key in list(dictionary.keys()) :
		value = dictionary[key]
		new_value = new_values.get(value, -1)
		if new_value == -1 :
			new_values[value] = count
			new_value = count
			count += 1
		ret[key] = new_value
	return ret

def induced_graph(partition, graph):
	"""Produce the graph where nodes are the communities """
	ret = nx.Graph()
	ret.add_nodes_from(list(partition.values()))

	for node1, node2, datas in graph.edges_iter(data = True) :
		weight = datas.get("weight", 1)
		com1 = partition[node1]
		com2 = partition[node2]
		w_prec = ret.get_edge_data(com1, com2, {"weight":0}).get("weight", 1)
		ret.add_edge(com1, com2, weight = w_prec + weight)
	return ret

def __neighcom(node, graph, status) :
	""" Compute the communities in the neighborood of node in the graph given
    with the decomposition node2com
	"""
	weights = {}
	for neighbor, datas in graph[node].items() :
		if neighbor != node :
			weight = datas.get("weight", 1)
			neighborcom = status.node2com[neighbor]
			weights[neighborcom] = weights.get(neighborcom, 0) + weight

	return weights


def __remove(node, com, weight, status) :
	""" Remove node from community com and modify status"""
	status.degrees[com] = ( status.degrees.get(com, 0.) - status.gdegrees.get(node, 0.) )
	status.internals[com] = float( status.internals.get(com, 0.) - weight - status.loops.get(node, 0.) )
	status.node2com[node] = -1


def __insert(node, com, weight, status) :
	""" Insert node into community and modify status"""
	status.node2com[node] = com
	status.degrees[com] = ( status.degrees.get(com, 0.) + status.gdegrees.get(node, 0.) )
	status.internals[com] = float( status.internals.get(com, 0.) + weight + status.loops.get(node, 0.) )


def __one_level(graph, status) :
	""" Compute one level of communities """
	modif = True
	nb_pass_done = 0
	cur_mod = __modularity(status)
	new_mod = cur_mod

	while modif  and nb_pass_done != __PASS_MAX :
		cur_mod = new_mod
		modif = False
		nb_pass_done += 1

		for node in graph.nodes() :
			com_node = status.node2com[node]
			degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight*2.)
			neigh_communities = __neighcom(node, graph, status)
			__remove(node, com_node, neigh_communities.get(com_node, 0.), status)
			best_com = com_node
			best_increase = 0
			for com, dnc in neigh_communities.items() :
				incr =  dnc  - status.degrees.get(com, 0.) * degc_totw
				if incr > best_increase :
					best_increase = incr
					best_com = com                    
			__insert(node, best_com, neigh_communities.get(best_com, 0.), status)
			if best_com != com_node : modif = True                
		new_mod = __modularity(status)
		if new_mod - cur_mod < __MIN : break

def generate_dendogram(graph, part_init = None):
	""" Finding the communities in the graph and return the associated dendogram """
	current_graph = graph.copy()
	status = Status()
	status.init(current_graph, part_init)
	mod = __modularity(status)
	status_list = list()
	__one_level(current_graph, status)
	new_mod = __modularity(status)
	partition = __renumber(status.node2com)
	status_list.append(partition)
	mod = new_mod
	current_graph = induced_graph(partition, current_graph)
	status.init(current_graph)

	while True :
		__one_level(current_graph, status)
		new_mod = __modularity(status)
		if new_mod - mod < __MIN : break
		partition = __renumber(status.node2com)
		status_list.append(partition)
		mod = new_mod
		current_graph = induced_graph(partition, current_graph)
		status.init(current_graph)
	return status_list[:]

def partition_at_level(dendogram, level) :
	""" Return the partition of the nodes at the given level """
	partition = dendogram[0].copy()
	for index in range(1, level + 1) :
		for node, community in partition.items() :
			partition[node] = dendogram[index][community]
	return partition

def best_partition(graph, partition = None):
	dendo = generate_dendogram(graph, partition)
	return partition_at_level(dendo, len(dendo) - 1)


def readannotationfile(FileName):
	""" 
		Reading protein file and constructing protgo dictionary from the database, it takes 3 parameters and return 2 values:
		Inputs: FileName is the name of file containing protein pairs, Id is the protein Identifiers 1 for UniProt Ids and 2 for Genenames
				  and so indicating the ontology undeer consideration!
		Outputs: 
	"""
	global protgo, icc, termcc, Head

	fp = open(os.path.join(os.path.dirname(__file__), FileName), 'r')
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

def readtargets(FileName):
	""" 
		Reading the protein GO annotation file to protgo dictionary used to identify proteins fuzzy annotated to go terms
	"""
	termset = set()
	fp = open(FileName)
	for line in fp:
		ligne = line.strip()
		if not ligne: continue
		ligne = ligne.split()
		term = ligne[0].strip()
		termset.add(term)
		del term
	fp.close()
	return list(termset)

def _fixkwargs(dicts, allapp):
	"""
		This module checks different parameters *args and **kwargs and align them correctly in order to run funcsim module.
	"""
	if len(dicts) > 7: 
		print("7 or 8 arguments are required but more than 8 arguments provided.\nFor now, the process cannot be pursued: number of arguments error ...") 
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
		sys.exit(2)

	if 'measure' not in dicts: kwargs['measure'] = 'ubma'
	elif type(dicts['measure'])==str: 
		if dicts['measure'].lower() in allapp: kwargs['measure'] = dicts['measure'].lower()
		else:
			print("Check notations of different measures.\nFor now, the process cannot be pursued: measure key error ...")
			sys.exit(3)	
	else:
		print("Measure key is either a symbol of measure.\nCheck measure symbols and try again ...")
		sys.exit(4)

	if 'score' not in dicts: kwargs['score'] = 0.3
	elif type(dicts['score'])==float and dicts['score'] >= 0.0: kwargs['score'] = float(dicts['score'])
	else:
		print("Check your semantic similarity score cut-off, it must be a real number: 0 < score <= 1.\n\nFor now, the process cannot be pursued: Cut-off score value error ...")
		sys.exit(5)

	if 'mclust' not in dicts: kwargs['mclust'] = 1
	elif isinstance(dicts['mclust'], int) and 1 <= dicts['mclust'] <= 3: kwargs['mclust'] = dicts['mclust']
	else:
		print("Clustering approach must be an integer 1, 2 or 3.\n\t1. For hierarchical model\n\t2. Graph spectral or kmean model\n\t3. Heuristic-based model")
		sys.exit(6)

	if 'nclust' not in dicts: kwargs['nclust'] = 0
	elif isinstance(dicts['nclust'], int) and dicts['nclust'] >= 0: kwargs['nclust'] = dicts['nclust']
	else:
		print("The number of clusters must be a positive integer.\n\nThe process cannot be pursued: number of clusters value error ...")
		sys.exit(7)

	if 'drop' not in dicts: kwargs['drop'] = 0
	elif dicts['drop'] is 0 or dicts['drop'] is 1: kwargs['drop'] = dicts['drop']
	else:
		print("Check the use of IEA evidence code variable <drop> which should be a Boolean:\n\t0 if all evidence code should be used and \n\t1 if IEA evidence code should be excluded.\n\nPlease check and try again ...") # raise ValueError
		sys.exit(8)

	if 'output' not in dicts: kwargs['output'] = 1
	elif dicts['output'] is 0 or dicts['output'] is 1 or dicts['output'] is 2: kwargs['output'] = dicts['output']
	else:
		print("How do you want to output results is an Enum 0, 1, 2:\n\t1 if results should be displayed on the screen \n\t0 if results should be written in a file.\n\t2 for outputting a Python object for possible further usage.\n\nPlease check and try again ...") # raise ValueError
		sys.exit(9)
	kwargs['tablefmt'] = 'rst' # One can also use 'grid' or other display formats!
	return kwargs

def proteinfct(*args, **kwargs):
	"""
proteinfct: protein functional clustering tool
This function from ProteinClustering.py allows the partitioning of a 
gene or protein set into a set of biological meaningful sub-classes 
using their functional closeness based on GO annotations and derived 
from a selected semantic similarity model. 

*args* is a variable length argument, which is mainly a string 
representing the name of the file containing protein IDs and their 
associated GO IDs and possibly the list of proteins/genes for which 
functional similarity (SS) scores must be retrieved.

*kwargs* can be used to set: ontology, measure, drop, output, mclust
and nclust.

Usage:
------
	This function requires at least two arguments and takes up to nine
	arguments.
	(a) Dictionary of protein/gene and  or name of the file containing 
	protein/genes and associated GO IDs annotating them.
	(b) Protein or gene ID or a list/tuple or the ontology under conside-
	ration
	(c) For the functional similarity measure, refer to funcsim (see 3.1)
	(d) The threshold score providing the functional similarity degree at 
	which proteins can be considered to be functionally close, set to 0.3
	by defaut. The score 0 indicates that all protein pairs with func-
	tional similarity score greater that 0.
	(d) drop and output as described previously.  
	(e) Clustering model under consideration, and this function implements
	three different models (mclust):
  		- hierarchical clustering (mclust = 1)
  		- Graph spectral clustering or kmeans (mclust = 2)
  		- community detecting model by Thomas Aynaud, 2009 (mclust = 3)
	(f) Number of clusters (nclust) applies only for the kmeans model and
	it is set to 0 by default. In this case, if mclust is less than 2 then
	the comminity detecting model is applied instead of kmeans!

  		>>> proteinfct(AnnotationData, Targets=[], ontology='BP', measure='ubma', score=0.3, mclust=1, nclust=0, drop=0, output=1)

Examples:
---------
  		>>> from dagofun.ProteinClustering import *
  		>>> proteinfct('tests/TestProteins.txt', measure='nbma', score=0.0, mclust=2, nclust=3)
  		>>> proteinfct('tests/TestProteins.txt', mclust=2)
  		>>> proteinfct('tests/TestProteins.txt', measure='rbmm', score=0.0, mclust=1)
  		>>> background = {'Q5H9L2':['GO:0006355','GO:0006351'], 'P03891':['GO:0022904','GO:0044281','GO:0044237','GO:0006120'], 'Q5H9L2':['GO:0006355','GO:0006351']}
		>>> proteinfct(background, measure='rbmm', score=0.0)
		>>> proteinfct('tests/SpecificRefSet2.txt', measure='ubmm', score=0.0, mclust=3)
		>>> proteinfct('tests/SpecificRefSet1.txt', score=0.0)
	"""
	print("\n************************************************************************************************************")
	print("       Package A-DaGO-Fun: A General Gene Ontology Semantic Similarity based Functional Analysis Tool")
	print("           Computational Biology Group (CBIO) & African institute for Mathematical Sciences (AIMS)")
	print("                        Distributed under free software (GNU General Public Licence) ")
	print("                              (c) 2015 GPL, Verson 15.1, All rights reserved.")
	print("************************************************************************************************************\n")
	global Ont, icc, termcc, TermSim, protgo, MaxValue
	Fsim = {'agic':(1,1),'adic':(1,2),'auic':(1,3),'acou':(1,4), 'acot':(1,5), 'xravg':(1,4,10),'xrbma':(1,4,11),'xrabm':(1,4,12),'xrbmm':(1,4,13), 'xrhdf':(1,4,14), 'xrvhdf':(1,4,15), 'xrmax':(1,4,16),'xnavg':(1,5,10),'xnbma':(1,5,11),'xnabm':(1,5,12), 'xnbmm':(1,5,13), 'xnhdf':(1,5,14), 'xnvhdf':(1,5,15),'xnmax':(1,5,16),'xlavg':(1,6,10),'xlbma':(1,6,11),'xlabm':(1,6,12),'xlbmm':(1,6,13), 'xlhdf':(1,6,14), 'xlvhdf':(1,6,15),'xlmax':(1,6,16),'ravg':(1,7,10),'rbma':(1,7,11),'rabm':(1,7,12),'rbmm':(1,7,13), 'rhdf':(1,7,14), 'rvhdf':(1,7,15),'rmax':(1,7,16),'navg':(1,8,10),'nbma':(1,8,11),'nabm':(1,8,12),'nbmm':(1,8,13), 'nhdf':(1,8,14), 'nvhdf':(1,8,15),'nmax':(1,8,16),'lavg':(1,9,10),'lbma':(1,9,11),'labm':(1,9,12),'lbmm':(1,9,13), 'lhdf':(1,9,14), 'lvhdf':(1,9,15),'lmax':(1,9,16),'savg':(1,10,10),'sbma':(1,10,11),'sabm':(1,10,12),'sbmm':(1,10,13), 'shdf':(1,10,14), 'svhdf':(1,10,15),'smax':(1,10,16),'liavg':(1,11,10),'libma':(1,11,12),'liabm':(1,11,12),'libmm':(1,11,13), 'lihdf':(1,11,14), 'livhdf':(1,11,15),'limax':(1,11,16),'ugic':(2,1),'udic':(2,2),'uuic':(2,3),'ucou':(2,4), 'ucot':(2,5),'uavg':(2,1,10),'ubma':(2,1,11),'uabm':(2,1,12),'ubmm':(2,1,13), 'uhdf':(2,1,14), 'uvhdf':(2,1,15),'umax':(2,1,16),'wgic':(3,1),'wdic':(3,2),'wuic':(3,3),'wcou':(3,4), 'wcot':(3,5),'wavg':(3,2,10),'wbma':(3,2,11),'wabm':(3,2,12), 'wbmm':(3,2,13), 'whdf':(3,2,14), 'wvhdf':(3,2,15),'wmax':(3,2,16),'zgic':(4,1),'zdic':(4,2),'zuic':(4,3),'zcou':(4,4), 'zcot':(4,5),'zavg':(4,3,10),'zbma':(4,3,11),'zabm':(4,3,12),'zbmm':(4,3,13), 'zhdf':(4,3,14), 'zvhdf':(4,3,15),'zmax':(4,3,16),'ui':(1,6), 'ub':(1,7), 'db':(1,8), 'nto':(1,9)}
	appnames = {'agic':'Annotation-based SimGIC','adic':'Annotation-based SimDIC','auic':'Annotation-based SimGUIC','xravg':'XGraSM-Resnik based Average','xrbma':'XGraSM-Resnik based Best Match Average','xrabm':'XGraSM-Resnik based Averaging Best Matches','xrmax':'XGraSM-Resnik based Maximum','xnavg':'XGraSM-Nunivers based Average','xnbma':'XGraSM-Nunivers based Best Match Average','xnabm':'XGraSM-Nunivers based Averaging Best Matches','xnmax':'XGraSM-Nunivers based Maximum','xlavg':'XGraSM-Lin based Average','xlbma':'XGraSM-Lin based Best Match Average','xlabm':'XGraSM-Lin based Averaging Best Matches','xlmax':'XGraSM-Lin based Maximum','ravg':'Resnik-based Average','rbma':'Resnik-based Best Match Average','rabm':'Resnik-based Averaging Best Matches','rmax':'Resnik-based Average','navg':'Nunivers-based Average','nbma':'Nunivers-based Best Match Average','nabm':'Nunivers-based Averaging Best Matches','nmax':'Nunivers-based Maximum','lavg':'Lin-based Average','lbma':'Lin-based Best Match Average','labm':'Lin-based Averaging Best Matches','lmax':'Lin-based Maximum','savg':'Relevance-based Average','sbma':'Relevance-based Best Match Average','sabm':'Relevance-based Averaging Best Matches','smax':'Relevance-based Maximum','liavg':'Li-based Average','libma':'Li-based Best Match Average','liabm':'Li-based Averaging Best Matches','limax':'Li-based Maximum','ugic':'GO-universal based SimGIC','udic':'GO-universal based SimDIC','uuic':'GO-universal based SimGIC','uavg':'GO-universal based Average','ubma':'GO-universal based Best Match Average','uabm':'GO-universal based Averaging Best Matches','umax':'GO-universal based Maximum','wgic':'Wang et al. based SimGIC','wdic':'Wang et al. based SimDIC','wuic':'Wang et al. based SimGUIC','wavg':'Wang et al. based Average','wbma':'Wang et al. based Best Match Average','wabm':'Wang et al. based Averaging Best Matches','wmax':'Wang et al. based Maximum','zgic':'Zhang et al. based SimGIC','zdic':'Zhang et al. based SimDIC','zuic':'Zhang et al. based SimGUIC','zavg':'Zhang et al. based Average','zbma':'Zhang et al. based Best Match Average','zabm':'Zhang et al. based Averaging Best Matches','zmax':'Zhang et al. based Maximum','ui':'Union-Intersection based','rbmm':'Resnik based Best Match Maximim','rhdf':'Resnik-based derived from the Hausdorff metric','rvhdf':'Resnik-based derived from a variant Hausdorff measure','xrbmm':'XGraSM-Resnik based Best Match Maximum','xrhdf':'XGraSM-Resnik-based derived from the Hausdorff metric','xrvhdf':'XGraSM-Resnik-based derived from a Variant Hausdorff measure','xnbmm':'XGraSM-Nunivers based Best Match Maximim','xnhdf':'XGraSM-Nuvivers-based derived from the Hausdorff metric','xnvhdf':'XGraSM-Nunivers-based derived from a Variant Hausdorff measure','nbmm':'NUnivers based Best Match Maximum','nhdf':'NUnivers-based derived from the Hausdorff metric','nvhdf':'Nunivers-based derived from a Variant Hausdorff measure','lbmm':'Lin based Best Match Maximum','lhdf':'Lin-based derived from the Hausdorff metric','lvhdf':'Lin-based derived from a variant Hausdorff','xlbmm':'XGraSM-Lin based Best Match Maximim','xlhdf':'XGraSM-Lin-based derived from the Hausdorff metric','xlvhdf':'XGraSM-Lin-based derived from a variant Hausdorff measure','sbmm':'Relevance based Best Match Maximum','shdf':'Relevance-based derived from the Hausdorff metric','svhdf':'Relevance-based derived from a variant Hausdorff measure','lihdf':'Li et al.-based derived from the Hausdorff metric','livhdf':'Li et al.-based derived from a variant Hausdorff measure','libmm':'Li et al. based Best Match Maximum','acou':'Annotation-based SimCOU','acot':'Annotation-based SimCOT','ubmm':'GO-Universal based Best Match Maximum','uhdf':'GO-universal based functional similarity derived from the Hausdorff metric','uvhdf':'GO-universal based functional similarity derived from a variant Hausdorff measure','ucou':'GO-universal-based SimCOU','ucot':'GO-universal-based SimCOT','wbmm':'Wang et al. based Best Match Maximum','whdf':'Wang et al. based functional similarity derived from the Hausdorff metric','wvhdf':'Wang et al. based functional similarity derived from a variant Hausdorff measure','wcou': 'Wang et al. based SimCOU','wcot':'Wang et al. based SimCOT','zbmm':'Zhang et al. based Best Match Maximum','zhdf':'Zhang et al. based functional similarity derived from the Hausdorff metric','zvhdf':'Zhang et al. based functional similarity derived from a variant Hausdorff measure','zcou':'Zhang et al. based SimCOU','zcot':'Zhang et al. based SimCOT','ub':'Universal-based','db':'Dice-based','nto':'Normalized Term Overlap based','ub':'Universal-based'}
	sdpt = -1 # Indicate the source of data: from a dictionary read 1 or from a file 0 and 2 for a two set GO IDs.
	
	now = time.time()
	print("Start processing your request on %s\n"%str(time.asctime(time.localtime())))
	
	# Checking different parameters ...
	if len(args) < 1:
		print('Illegal number of arguments: at least one argument required, no argument given.\nFor now, the process cannot be pursued: number of arguments error ...')
		sys.exit(1)
	elif  len(args)+len(kwargs) > 8:
		print('Illegal number of arguments: at most 6 arguments required, more than 6 arguments given.\nFor now, the process cannot be pursued: number of arguments error ...')
		sys.exit(2)
	elif len(args)==1: # This means that the only argument is a file containing GO ID pairs
		if type(args[0])==dict: sdpt = 1
		elif os.path.exists(os.path.join(os.path.dirname(__file__), args[0])): sdpt = 0
		else: # raise IOError
			print("The file or object provided cannot be opened for reading.\nThere is argument value error. Check and try again ...")
			sys.exit(3)
		kwargs = _fixkwargs(kwargs, Fsim)
	elif 2<=len(args)<=9: # This means that arguments are protein/gene pairs or a file/dictionary and ontology
		if not os.path.exists(os.path.join(os.path.dirname(__file__), args[0])) and type(args[0])!=dict:
			print ("Wrong format of Annotation data is detected or\nthere is argument value error. Check and try again ...")
			sys.exit(4)
		if not os.path.exists(os.path.join(os.path.dirname(__file__), args[1])) and not isinstance(args[1],(list, tuple, str)):
			print ("Wrong format of protein targets is detected or\nthere is argument value error. Check and try again ...")
			sys.exit(5)
		if len(args)>=3: 
			kwargs['ontology'] = args[2]
			if len(args)>=4: 
				kwargs['measure'] = args[3]
				if len(args)>=5: 
					kwargs['score'] = args[4]
					if len(args)>=6: 
						kwargs['mclust'] = args[5]
						if len(args)>=7: 
							kwargs['nclust'] = args[6]
							if len(args)>=8: 
								kwargs['drop'] = args[7]
								if len(args)>=9: kwargs['output'] = args[8]
		kwargs = _fixkwargs(kwargs, Fsim)
	else:
		print("There is inconsistency in your input parameters. Please check different parameters provided and try again ...") 
		sys.exit(6)
	
	SOntology = 1 + Ont.index(kwargs['ontology'])
	
	# Loading GO data
	getGOdata(SOntology)
	
	# Loading Annotation data and protein targets (candidate for clustering)
	try:
		readannotationfile(args[0])  # Reading reference or background dataset
	except:
		if type(args[0])==dict:
			for p in args[0]: 
				protgo[p] = set([icc[t] for t in args[0][p] if (t in icc) and termcc[icc[t]][-1] and icc[t]!=Head])
				if not protgo[p]: protgo.pop(p, None) # Delete safely the concept added when it does not have annotations
		else:
			print ("Check the file containing of Annotation data or the format of annotation data and try again ...")
			sys.exit(7)
	if not protgo:
		print("Possibly GO IDs in the annotation file are not found in the version of GO dataset used.\nThe process cannot be pursued ...")
		sys.exit(8)

	targets = []
	if len(args)>=2:
		try:
			targets = readtargets(args[1])
		except:
			if isinstance(args[1], (tuple, list, set)): targets = set(args[1])
			else:
				print("Check the file containing of target GO IDs or the format of target GO IDs data and try again ...")
				sys.exit(6)

	if targets:
		pp = list(set(targets) & set(protgo.keys()))
	else:
		pp = list(protgo.keys())
	ppairs = [(pp[i],pp[j]) for i in range(len(pp)) for j in range(i+1,len(pp))] # Get protein pairs
	
	getICdata(SOntology, Fsim[kwargs['measure']][0], kwargs['drop'])
	data = {}  # Computing Protein similarity Scores!
	if len(Fsim[kwargs['measure']])==3:      	# Corresponding to the term pair based protein functional similarity 
		if Fsim[kwargs['measure']][1] in [4, 7] and not MaxValue: MaxValue = max(goic.values())
		for p in ppairs:
			if p[0] in protgo and p[1] in protgo:
				retrieveTermSimilarity(p, Fsim[kwargs['measure']][1])
				data[p] = computingProteinSimilarityScore(p, Fsim[kwargs['measure']][-1])
	elif len(Fsim[kwargs['measure']])==2:
		for p in ppairs:
			if p[0] in protgo and p[1] in protgo: data[p] = computingProteinSimilarityScore(p, Fsim[kwargs['measure']][-1])
	
	# CLustering process starts here
	g = nx.Graph(); agree = kwargs['score']
	for p in data: # Constructing graph goes here!
		if agree==0.0 and data[p] > 0.0: 
			g.add_edge(p[0], p[1], weight=1.0-data[p])
		elif 0 < agree < 1.0 and data[p] >= agree: g.add_edge(p[0], p[1], weight = 1.0-data[p])
		elif agree==1.0 and data[p]==1.0: g.add_edge(p[0], p[1], weight = 1.0)
	models = ['Hierarchical', 'Graph spectral based (kmeans)', 'Model-based']
	if g:
		# Outputting different results
		print("Clustering proteins based on functional similarity using:", "[proteinfct function from ProteinClustering.py]")
		print("# The number of possible proteins/genes and protein/gene pairs detected are %d and %d, respectively."%(len(g),g.size()))
		print("The distance is based on functional similarity measure  :", appnames[kwargs['measure']])
		print("The clustering model used is                            :", models[kwargs['mclust']-1], 'approach')
		if kwargs['output']:
			print("\nDifferent clusters are displayed in the table below or in the following figure.\nIf possible, use full screen mode for more convenient visualization:")
		else:
			inputdata = inspect.getframeinfo(inspect.currentframe().f_back)[3][0].strip().split(',')[0].split('(')[1].strip()
			if kwargs['mclust']==1:
				outputfile = inputdata.replace('\'','').replace('\"','').replace(':','_').split('.')[0].split('/')[-1].strip()+'Cl.png'	
			else: outputfile = inputdata.replace('\'','').replace('\"','').replace(':','_').split('.')[0].split('/')[-1].strip()+'Cl.txt'
			print("Different clusters can be found in the file       : [%s]"%(outputfile,))
		classes = {}
		if kwargs['mclust']==3:
			partition = best_partition(g); j = 0
			for i in set(partition.values()):
				classes[j] = [nodes for nodes in list(partition.keys()) if partition[nodes] == i]
				j += 1
		elif kwargs['mclust']==1: # Hierarchical approach
			Index = g.nodes()
			n = len(Index)
			distances = zeros((n,n))
			path_length = nx.all_pairs_dijkstra_path_length(g)	
			for u,p in path_length.items():
				for v,d in p.items():
					distances[Index.index(u)][Index.index(v)] = d
					distances[Index.index(v)][Index.index(u)] = d
			sd = distance.squareform(distances)
			hier = hierarchy.average(sd)
			fig = plt.figure()
			hierarchy.dendrogram(hier, orientation='right', labels=Index[:], leaf_font_size=1)
			plt.grid()
			if kwargs['output']==1:
				print("\nWarning:: In case the module is being run from a server, please choose 0 or 2 to write \nthe figure produced in a file from the server.")
				plt.show()
			else:
				print("\nWarning:: For the hierarchical approach, 0 or 2 produce the figure produced in a file\nfor possible further usage.")
				plt.savefig(outputfile, format="png")
		elif kwargs['mclust']==2: # Graph spectral based (kmeans)
			d =  kwargs['nclust']; classes = {}# Number of presumed clusters
			if d < 2:
				partition = best_partition(g); j = 0
				for i in set(partition.values()):
					classes[j] = [nodes for nodes in list(partition.keys()) if partition[nodes] == i]
					j += 1
			else:
				W = nx.adj_matrix(g)
				D = diag([sum(sum(array(w))) for w in W])
				L = D - W
				S, V, D = svd(L)
				N = g.nodes()
				test = True; d = 2
				while test:
					cidx = []
					res, idx = kmeans2(S[:,-d+1:], d, minit='random') #-d+1:-1
					cidx = unique(idx)
					classes = {}
					for i in range(len(N)):
						if idx[i] in classes: classes[idx[i]].append(N[i])
						else: classes[idx[i]] = [N[i]]

					k = 0 # Checking if nodes in the identified class are connected!
					for iclass in classes.values():
						if not nx.is_connected(nx.subgraph(g,iclass)):
							k = 1
							break
					if not k: test = False
		if classes:
			outs = []
			for i in sorted(classes.keys()):
				st = str(classes[i])[1:-1]
				outs.append(('%d'%(i+1,), '%d'%(len(classes[i]),), st[:78]))
				for i in range(1, int(ceil(len(st)/78.0))):
					try: outs.append(('','',st[i*78:(i+1)*78+1]))
					except: pass
				outs.append(('','','\n'))
			if kwargs['output']==2:
				print("\nDiscovering functionally similar proteins is accomplished on %s"%str(time.asctime(time.localtime())))
				print("Total time elapsed is approximately:", (time.time()-now), 'seconds\n')
				print("\n************************************************************************************************************\n")
				return outs 
			headers = ['# Cluster', '# of Proteins', 'Protein identifiers']
			if kwargs['output']: # Print on the screen
				print('%s'%tab(outs[:-1], headers, kwargs['tablefmt'], floatfmt=".5f", stralign="left"))
			else:
				try:
					fp = open(outputfile, 'w')
					fp.write("# Clustering proteins based on functional similarity using [%s] approach\n"%(models[kwargs['mclust']-1],))
					fp.write("# The number of possible proteins/genes and protein/gene pairs detected are %d and %d, respectively.\n"%(len(g),g.size()))
					fp.write("# The distance is based on [%s] functional similarity measure\n\n"%(appnames[kwargs['measure']],))
					fp.write('%s'%tab(outs[:-1], headers, kwargs['tablefmt'], floatfmt=".5f", stralign="left"))
					fp.close()
				except IOError:
					print("File cannot be opened in writing. Check possibly the writing permission and try again ...")
					sys.exit(8)
	else:
		print('Trying %s clustering approach using the distance inferred from\n %s functional similarity measure has failed. Please, check your presumed\n list and options you have selected, and try again!'%(models[kwargs['mclust']-1], appnames[kwargs['measure']]))
		sys.exit(9)
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

	python $(python -m site --user-site)/dagofun/ProteinClustering.py AnnotationData ProteinTargets ontology measure score mclust nclust drop output

Different arguments are as explained in the termsim function (see pa-
ckage documentation). 
Moreover these arguments should be in order as shown in the command and 
as for commands under a Python interpreter, AnnotationData file con-
taining the user input list of proteins and their associated GO IDs and 
ProteinPairs is another user input file containing protein IDs to be 
clustered, and these two files must be provided. 
In case where other parameters are not provided, default parameters are 
used.

Assuming that the package was not installed, then:

$(python -m site --user-site)/dagofun/

should be replaced by the path to modules.
	"""
	if len(argv) <= 2:
		print('\nIllegal number of arguments: at least one argument required, no argument given.\nFor now, the process cannot be pursued: number of arguments error ...\n')
	elif  len(argv) > 10:
		print('\nIllegal number of arguments: at most 10 argument required, more than 4 arguments given.\nFor now, the process cannot be pursued: number of arguments error ...\n')
	else:
		if len(argv)==3: proteinfct(argv[1].strip(), argv[2].strip())
		elif len(argv)==4: proteinfct(argv[1].strip(), argv[2].strip(), ontology = argv[3].strip())
		else: # measure score mclust nclust drop output
			try:
				Measure = 'ubma'; Score = 0.3; Mclust = 1; Nclust = 0; Drop = 0; Output = 1
				j = 4
				if len(argv) > j:
					Measure = argv[j].strip(); j += 1
					if len(argv) > j:
						try: 
							Score = float(argv[j].strip()); j += 1
							if not (0.0 <= Score <= 1.0):
								print("\nThere is inconsistency. Please refer to the package documentation,\ncheck score argument provided, which should be a float in [0,1] interval ...\n")
								return
						except:
							print("\nThere is inconsistency. Please refer to the package documentation,\ncheck score argument provided, which should be a float in [0,1] interval ...\n")
							return
						if len(argv) > j:
							try:
								Mclust = int(argv[j].strip()); j += 1
								if not (1 <= Mclust <= 3): 
									print("\nClustering approach mclust must be an integer 1, 2 or 3.\n\t1. For hierarchical model\n\t2. Graph spectral or kmean model\n\t3. Heuristic-based model\nCheck your mclust argument and try agian ...\n")
									return
							except:
								print("\nThere is inconsistency. Please refer to the package documentation,\ncheck mclust argument provided, which should be an Enum 1, 2 or 3 ...\n")
								return
							if len(argv) > j:
								try:
									Nclust = int(argv[j].strip()); j += 1
									if Nclust < 0:
										print("The number of clusters must be a positive integer.\n\nThe process cannot be pursued: number of clusters value error ...")
										return
								except:
									print("\nThere is inconsistency. Please refer to the package documentation,\ncheck nclust argument provided, which should be an integer >= 0 ...\n")
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
												print("\nHow do you want to output results should be an Enum 0, 1, 2:\n\t1 if results should be displayed on the screen \n\t0 if results should be written in a file.\n\t2 for outputting a Python object for possible further usage.\n\nRunning this module using a command line is plausible only when\nthe output parameter is set to 0 or 1.\nPlease check and try again ...\n")
												return
										except:
											print("\nThere is inconsistency. Please refer to the package documentation,\ncheck drop parameter provided ...\n")
											return
				proteinfct(argv[1].strip(), argv[2].strip(), ontology = argv[3].strip(), measure = Measure, mclust = Mclust, nclust = Nclust, drop = Drop, output = Output)
			except:
				print("\nThere is inconsistency. Please refer to the package documentation,\ncheck the nappr argument provided and try again ...\n")

if __name__=='__main__':
	_main(sys.argv)
