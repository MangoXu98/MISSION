'''
Calculate the similarity base on disease information network
'''
import argparse
import pandas as pd
import numpy as np
import logging
from pprint import pprint
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.sparse import csr_matrix as csr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import *
import time


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='myapp.log',
                    filemode='w')


def generate_adj_mat(edges, did2ind, tid2ind):
    '''
    Generate adjacent matrix
    '''
    data, rows, cols = [], [], []
    for edge in edges:
        data.append(1)
        rows.append(did2ind[edge[0]])
        cols.append(tid2ind[edge[1]])
    adj = csr((data, (rows, cols)), shape=[
              len(did2ind), len(tid2ind)]).toarray()
    adj_t = csr((data, (cols, rows)), shape=[
                len(tid2ind), len(did2ind)]).toarray()
    return adj, adj_t


def read_GraphData(args, target):
    '''
    Building a Graph from data sets
    '''
    print("=============== Reading Diseases Graph Data... ================")
    raw_data = pd.read_csv(args.datapath.format(target))
    d_id = np.unique(raw_data.iloc[:, 0])
    t_id = np.unique(raw_data.iloc[:, 1])
    edges = raw_data.iloc[:, [0, 1]].values
    print("The loading of graph data", target, "has completed...")
    return d_id, t_id, edges


def cal_structSim(did2ind, tid2ind_list, adj_list):
    '''
    calculate the similarity between 2 nodes by meta-structure
    '''
    logging.info('Calculating the structure similarity between nodes')
    sim = {}
    ind2did = reverse_map(did2ind)
    mimnumber_set = ind2did.keys()
    pairs = list(combinations(mimnumber_set, 2))
    chunks = partition(pairs, 90)
    future = {}

    with ProcessPoolExecutor(max_workers=16) as executor:
        part = 1
        for c in chunks:
            job = executor.submit(cal_StructSim_P, c,
                                  did2ind, tid2ind_list, adj_list)
            future[job] = part
            part += 1
        for job in as_completed(future):
            dl = job.result()
            sim.update(dl)
    return sim


def cal_StructSim_P(chunks, did2ind, tid2ind_list, adj_list):
    sim = {}
    ind2did = reverse_map(did2ind)
    for pair in tqdm(chunks):
        mim1 = pair[0]
        mim2 = pair[1]
        intersect_num_list = []
        path_count_self_mim1 = 0
        path_count_self_mim2 = 0
        for i in range(len(tid2ind_list)):
            tid2ind, adj = tid2ind_list[i], adj_list[i]
            intersect_num = np.dot(adj[mim1], adj[mim2])
            intersect_num_list.append(intersect_num)
            path_count_self_mim1 += np.sum(adj[mim1] == 1)
            path_count_self_mim2 += np.sum(adj[mim2] == 1)

        path_count_mim1_mim2 = 0
        for i in intersect_num_list:
            path_count_mim1_mim2 += i

        # calculate the structure similarity
        structsim_score = (2 * path_count_mim1_mim2) / \
            (path_count_self_mim1 + path_count_self_mim2)**0
        sim[(ind2did[mim1], ind2did[mim2])] = structsim_score
    logging.info('Calculating finish')
    return sim


def cal_Sim(d_id, t_id_list, edges_list):
    '''
    Calculate the similarity between two diseases
    '''
    print("=============== Processing data & Generating adjacent matri... ================")
    # disease id to index  & index to disease id
    did2ind = {v: k for k, v in enumerate(d_id)}
    ind2did = reverse_map(did2ind)

    tid2ind_list, adj_list = [], []
    ind2tid_list, adj_t_list = [], []

    for i in range(len(t_id_list)):
        # target id to index  & index to target id
        tid2ind = {v: k for k, v in enumerate(t_id_list[i])}
        ind2tid = reverse_map(tid2ind)
        # adjacent matrix and transposed adjacent matrix
        adj, adj_t = generate_adj_mat(edges_list[i], did2ind, tid2ind)
        tid2ind_list.append(tid2ind)
        ind2tid_list.append(ind2tid)
        adj_list.append(adj)
        adj_t_list.append(adj_t_list)
    logging.info('Processing finished')

    print('Calculating the similarity')

    # calculating the similarity between diseases
    sim = cal_structSim(did2ind, tid2ind_list, adj_list)
    print('Completed')
    return sim


def main(args):
    '''
        Calculate disease similarity based on DIN through meta structure
    '''
    print("=============== Calculate disease similarity based on DIN... ================")
    time_start = time.time()
    d_id, t_id_list, edges_list = [], [], []
    meta_structures = args.meta_structures
    sim = {}
    for meta_structure in meta_structures:
        print("Current meta structure:", meta_structure)
        for target in meta_structure:
            d_id, t_id, edges = read_GraphData(args, target)
            t_id_list.append(t_id)
            edges_list.append(edges)
        temp = cal_Sim(d_id, t_id_list, edges_list)
        for key in temp.keys():
            if key in sim.keys():
                sim[key] = max([sim[key], temp[key]])
            else:
                sim[key] = temp[key]
    print(len(list(sim.keys())))

    time_end_1 = time.time()
    print('total time cost for calculation of sim: ', time_end_1 - time_start)

    keys = sim.keys()
    disease1, disease2, similarity = [], [], []
    for i in keys:
        disease1.append(i[0])
        disease2.append(i[1])
        similarity.append(sim[i])
    logging.info('Writing the similarity between nodes')
    similarity = array_normalization(np.array(similarity))

    time_end_2 = time.time()
    print('total time cost for normalization of sim: ', time_end_2 - time_start)

    c = {'disease1': disease1, 'disease2': disease2, 'similarity': similarity}
    df = pd.DataFrame(c)
    df.to_csv(args.savepath, index=None)


if __name__ == '__main__':
    time_start = time.time()
    parser = argparse.ArgumentParser(description="Run In Meta Structure.")
    parser.add_argument('--datapath', type=str,
                        default='./dataset/associations/doid_{}_1754_1v1.csv')
    parser.add_argument('--targets', type=list,
                        default=['cheid', 'geneid', 'hpoid'])
    parser.add_argument('--savepath', type=str,
                        default='./runTime/DIN_sim.csv')
    parser.add_argument('--meta_structures', type=str,
                        default=[['hpoid', 'geneid'],['hpoid', 'cheid'],['cheid', 'geneid'],['hpoid', 'geneid', 'cheid']])
    args = parser.parse_args()

    main(args)

    time_end = time.time()
    print("\r\n Total time (s):", time_end - time_start)
