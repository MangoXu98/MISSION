'''
Calculate disease similarity based on disease annotations
'''
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from utils import *
from itertools import combinations
from tqdm import tqdm
import dagofun
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def cal_termIC():
    '''
    Calculate the IC value of GO terms
    '''
    print("=============== Calculate the IC value of GO terms... ================")
    df = pd.read_csv('dataset/annotations/GO_id_17920.csv')
    GO = df['DOID'].values.tolist()
    dagofun.getTermFeatures(GO, model='z', drop=0, output=0)


def termIC_process(termIC_path):
    '''
    Process txt file and store as csv
    '''
    print("=============== Process TXT file... ================")
    GO, IC_value = [], []
    txt_fileName = 'runTime/ann/GOTF_Zhang et al..txt'
    fopen = open(txt_fileName, 'r')
    lines = fopen.readlines()
    for line in tqdm(lines[13:20156]):
        line = line.strip('\n')
        sim_list = line.split()
        GOID = sim_list[0]
        if GOID.startswith('GO', 0, 2):
            GO.append(sim_list[0])
            if(sim_list[-1] == 'U'):
                IC_value.append(0.0)
            else:
                IC_value.append(float(sim_list[-1]))
    c = {'GO': GO, 'IC_value': IC_value}
    df = pd.DataFrame(c)
    df.to_csv(termIC_path, index=None)


def calc_pairs_sim(d2g_path, termIC_path):
    '''
    Calculate the similarity between disease pairs
    '''
    print("=============== Parallel calculate the similarity between disease pairs ================")
    data_all = pd.read_csv(d2g_path).set_index('DOID')['GOID']
    data =  data_all.to_dict()

    pairs = list(combinations(list(data.keys()), 2))

    go2IC_dict = pd.read_csv(termIC_path).set_index('GO')['IC_value'].to_dict()
    chunks = partition(pairs, 100)

    future = {}
    time_start = time.time()
    df = pd.DataFrame({'disease1': [], 'disease2': [], 'similarity': []})

    with ProcessPoolExecutor(max_workers=16) as executor:
        part = 1
        for c in chunks:
            job = executor.submit(syc, data, c, go2IC_dict)
            future[job] = part
            part += 1
        for job in as_completed(future):
            temp = job.result()
            df = pd.concat([df, temp])
    time_funcsim = time.time()
    print('total time cost for calculation of disease pairs: ',
          time_funcsim - time_start)
    df.to_csv('./runTime/ann_similarity.csv', index=None)


def syc(data, pairs, go2IC_dict):
    '''
    Parallel Computing
    '''
    disease1, disease2, similarity = [], [], []
    for pair in tqdm(pairs):
        if str(data[pair[0]]) != 'nan' and str(data[pair[1]]) != 'nan':
            set1 = eval(data[pair[0]])
            set2 = eval(data[pair[1]])
            go_list = list(set1 | set2)
            vec1 = np.zeros(len(go_list))
            vec2 = np.zeros(len(go_list))
            for i in set1:
                index = go_list.index(i)
                if i in go2IC_dict.keys():
                    vec1[index] = go2IC_dict[i]

            for i in set2:
                index = go_list.index(i)
                if i in go2IC_dict.keys():
                    vec2[index] = go2IC_dict[i]
            sim = cos_sim(vec1, vec2)
        else:
            sim = 0

        disease1.append(pair[0])
        disease2.append(pair[1])
        similarity.append(sim)
    df3 = pd.DataFrame(
        {'disease1': disease1, 'disease2': disease2, 'similarity': similarity})
    return df3


if __name__ == '__main__':
    time_start = time.time()
    d2g_path = './dataset/annotations/doid_goid_1754.csv'

    termIC_path = './runTime/ann/GOTF_Zhang_20..csv' 

    cal_termIC()
    termIC_process(termIC_path)

    calc_pairs_sim(d2g_path, termIC_path)
    time_end = time.time()
    print("\r\n Total time (s):", time_end - time_start)
