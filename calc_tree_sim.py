'''
Calculate the similarity between diseases based on the disease taxonomy tree
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import pickle
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import *
import time


def get_dict_id2edge(path, T='sub'):
    '''
    process the file and return a dictionary of disease_id an its relationship in disease ontology
    '''
    print("=============== Process DO file... ================")
    if T == 'all':
        df = pd.read_csv(path, usecols=[0, 11, 5])
    else:
        df = pd.read_csv(path)
    # print(df.shape[0])
    disease_doid = np.array(df['id'])
    disease_edge = np.array(df['is_a'])
    p = re.compile('[\]\[\'\"]')
    id_edge_dict = {}
    for index in range(len(disease_edge)):
        if disease_edge[index] == disease_edge[index]:
            node = disease_doid[index]
            string = re.sub(p, '', disease_edge[index]).split(',')
            edge = [text.split(' ! ')[0] for text in string]
            edge = [item.strip() for item in edge]
            id_edge_dict[node] = edge
            for item in edge:
                if item not in disease_doid:
                    id_edge_dict[node] = ['root']
        else:
            node = disease_doid[index]
            id_edge_dict[node] = ['root']
    id_edge_dict['root'] = ['root']
    return id_edge_dict


def find_all_ancestor(node1, node2, DO_dict):
    '''
    find all ancestors of 2 nodes
    '''
    node1_ancestors = get_ancestor_set([node1], DO_dict)
    node2_ancestors = get_ancestor_set([node2], DO_dict)
    temp1 = []
    temp2 = []
    while True:
        if temp1 == node1_ancestors and temp2 == node2_ancestors:
            break
        else:
            temp1 = node1_ancestors
            temp2 = node2_ancestors
            node1_ancestors = get_ancestor_set(node1_ancestors, DO_dict)
            node2_ancestors = get_ancestor_set(node2_ancestors, DO_dict)
    return node1_ancestors, node2_ancestors


def find_common_ancestor(node1, node2, G):
    '''
    find the nearest common ancestor of two nodes
    '''
    node1_ancestors = get_ancestor_set([node1], G)
    node2_ancestors = get_ancestor_set([node2], G)
    while True:
        if not list(set(node1_ancestors) & set(node2_ancestors)):
            node1_ancestors = get_ancestor_set(node1_ancestors, G)
            node2_ancestors = get_ancestor_set(node2_ancestors, G)
        else:
            ancestors = list(set(node1_ancestors) & set(node2_ancestors))
            break
    return ancestors


def get_ancestor_set(nodes, G):
    '''
    get the 1 step ancestor of nodes
    '''
    node_ancestors = []
    node_ancestors.extend(nodes)
    try:
        for node in nodes:
            if not node in G.keys():
                node_ancestors.extend(['root'])
            else:
                node_ancestors.extend(list(G[node]))
        if len(nodes) == len(node_ancestors):
            return set(node_ancestors)
        else:
            return set(node_ancestors)
    except Exception as e:
        print(str(e))
        return nodes


def cal_similarity(DO_dict, dis_list):
    '''
    calculate the diseases similarities base Wang's
    '''
    print("=============== Parallel computing disease similarity... ================")
    futures = {}
    keys = DO_dict.keys()
    DO_pair = list(combinations(dis_list, 2))
    chunks = partition(DO_pair, 1)
    df = pd.DataFrame({'disease1': [], 'disease2': [], 'similarity': []})
    with ProcessPoolExecutor(max_workers=4) as executor:
        part = 1
        for c in chunks:
            job = executor.submit(ontology_similarity_w, c, DO_dict)
            futures[job] = part
            part += 1
        for job in as_completed(futures):
            temp = job.result()
            df = pd.concat([df, temp])

    df.to_csv('./runTime/ont_similarity.csv', index=None)
    return


def ontology_similarity_w(pairs, DO_dict):
    '''
    calculating the similarity base wang's method
    '''
    disease1, disease2, similarity = [], [], []
    for pair in tqdm(pairs):
        DA_1 = 1
        DA_2 = 1
        T1, T2 = find_all_ancestor(pair[0], pair[1], DO_dict)
        common_T = list(set(T1) & set(T2))
        depth1 = node_deep(DA_1, DO_dict)
        depth2 = node_deep(DA_2, DO_dict)
        DV_1, DV_2, D_common = 0, 0, 0
        alpha = 0.5
        for i in T1:
            depth = node_deep(i, DO_dict)
            DV_1 += alpha**(depth-depth1)
        for i in T2:
            depth = node_deep(i,  DO_dict)
            DV_2 += alpha**(depth-depth2)
        for i in common_T:
            depth = node_deep(i, DO_dict)
            D_common += alpha**(depth-depth1)+alpha**(depth-depth2)
        sim = D_common/(DV_1+DV_2)
        disease1.append(pair[0])
        disease2.append(pair[1])
        similarity.append(sim)
    c = {'disease1': disease1, 'disease2': disease2, 'similarity': similarity}
    df = pd.DataFrame(c, index=None)
    return df


def node_deep(node, DO_dict):
    '''
    get the depth of node
    '''
    depth = 1
    ancestors = get_ancestor_set([node], DO_dict)
    while not 'root' in ancestors:
        depth += 1
        ancestors = get_ancestor_set(ancestors, DO_dict)
    return depth


def union_ontology_similarity(path):
    '''
    Combine multiple disease similarity files into one file
    '''
    sim = pd.DataFrame()
    for file in os.listdir(path):
        df = pd.read_csv(path+file)
        sim = sim.append(df, ignore_index=True)
    sim.to_csv('./runTime/ont_similarity.csv', index=None)


if __name__ == '__main__':
    time_start = time.time()

    path = './dataset/tree/doid.obo.txt_all.csv'
    DO_ALL_dict = get_dict_id2edge(path, T='all')

    # Calculate disease similarity
    dis_filename = './dataset/associations/doid_names_1754.csv'
    dis_dict = pd.read_csv(dis_filename).set_index('DOID')['NAME'].to_dict()
    dis_list = list(dis_dict.keys())
    
    cal_similarity(DO_ALL_dict, dis_list)

    time_end = time.time()
    print("\r\n Total time (s):", time_end - time_start)
