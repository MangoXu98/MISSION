'''
Fusion
'''
import pickle
from calc_DIN_sim import *
from utils import *
import tensorflow as tf
from compact_bilinear_pooling import compact_bilinear_pooling_layer as cbpl
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
import argparse
import logging
import numpy as np

tf.enable_eager_execution()

def fusion_similarity(file1, file2, targets):
    '''
    Fusion of disease similarity between two multimodal information
    '''
    print("=============== Fusion multimodal information... ================")
    disease1, disease2, similarity = [], [], []
    info1 = pd.read_csv(file1)
    info1['similarity'] = array_normalization(info1['similarity']) # normalization
    dict_1 = info1.set_index(['disease1', 'disease2']).to_dict()

    info2 = pd.read_csv(file2)
    info2['similarity'] = array_normalization(info2['similarity']) # normalization
    dict_2 = info2.set_index(['disease1', 'disease2']).to_dict()

    for key in tqdm(dict_1['similarity'].keys()):
        disease1.append(key[0])
        disease2.append(key[1])
        if key in dict_2['similarity'].keys():
            sim = dict_1['similarity'][key]*dict_2['similarity'][key]
        elif (key[1], key[0]) in dict_2['similarity'].keys():
            sim = dict_1['similarity'][key] * \
                dict_2['similarity'][(key[1], key[0])]

        similarity.append(sim)
    c = {'disease1': disease1, 'disease2': disease2, 'similarity': similarity}
    df = pd.DataFrame(c)
    df.to_csv(targets, index=None)


def get_weight_adjacent(file1, file2):
    '''
    get the weight adjacent matrix
    '''
    matrix_list = []
    hash_id2index = {}
    literature = pd.read_csv(file1)
    dict_l = literature.set_index(['disease1', 'disease2']).to_dict()
    structure = pd.read_csv(file2)
    dict_s = structure.set_index(['disease1', 'disease2']).to_dict()
    count = 0
    keys = []
    for i in dict_l['similarity'].keys():
        keys.append(i[0])
        keys.append(i[1])
    for key in np.unique(keys):
        hash_id2index[key] = count
        count += 1
    arr_l = np.ones(shape=(count, count))
    for key in dict_l['similarity'].keys():
        x = hash_id2index[key[0]]
        y = hash_id2index[key[1]]
        arr_l[x][y] = dict_l['similarity'][key]
        arr_l[y][x] = dict_l['similarity'][key]
    matrix_list.append(arr_l)
    arr_s = np.ones(shape=(count, count))
    for key in dict_s['similarity'].keys():
        if key in dict_l['similarity'].keys():
            x = hash_id2index[key[0]]
            y = hash_id2index[key[1]]
            arr_s[x][y] = dict_s['similarity'][key]
            arr_s[y][x] = dict_s['similarity'][key]
    matrix_list.append(arr_s)
    return matrix_list, hash_id2index


def cos_sim(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def L2Norm(x):
    '''
    L2 regularization
    '''
    sum = np.sum(np.power(x, 2))
    return [i/sum for i in x]


def signedSqrt(x):
    return np.sqrt(np.abs(x)) * np.sign(x)


def MCBP_fusion(file1, file2, target, dims):
    '''
    Fusion of disease similarity features based on DIN and multimodal-information through MCBP
    '''
    print("=============== MCBP Fusion... ================")
    matrix_list, hash_id2index = get_weight_adjacent(file1, file2)
    num = len(matrix_list[0])
    matrix_a = np.vsplit(matrix_list[0], 1754)
    matrix_b = np.vsplit(matrix_list[1], 1754)

    result_3_t2 = tf.zeros([1, dims])
    for i in tqdm(range(num)):
        matrix_a_t = tf.convert_to_tensor(matrix_a[i], dtype='float32')
        matrix_b_t = tf.convert_to_tensor(matrix_b[i], dtype='float32')
        result_3 = cbpl(matrix_a_t, matrix_b_t, dims)
        result_3_1 = signedSqrt(result_3)
        result_3_2 = L2Norm(result_3_1)
        result_3_t2 = tf.concat((result_3_t2, result_3_2), 0)
    result_3_t2 = np.delete(result_3_t2, 0, axis=0)

    result = result_3_t2

    hash_index2id = reverse_map(hash_id2index)
    disease1, disease2, similarity = [], [], []

    dis_vec = {}
    dis = []

    for i in range(1754):
        x = result[i]
        dis_vec[hash_index2id[i]] = x
        dis.append(hash_index2id[i])

    dis_com = combinations(dis, 2)
    for dis1, dis2 in tqdm(dis_com):
        similarity.append(cos_sim(dis_vec[dis1], dis_vec[dis2]))
        disease1.append(dis1)
        disease2.append(dis2)
    similarity = array_normalization(np.array(similarity))
    c = {'disease1': disease1, 'disease2': disease2, 'similarity': similarity}
    df = pd.DataFrame(c)
    df.to_csv(target, index=None)


if __name__ == '__main__':
    time_start = time.time()
    
    # parameters
    parser = argparse.ArgumentParser(
        description="Run In similarity fusion.")
    parser.add_argument(
        '--file5', type=str, default='./similarity/final/Ontology_similarity_normal.csv')  # ONT
    parser.add_argument(
        '--file4', type=str, default='./similarity/final/Attribute_similarity_normal.csv')  # ATT
    parser.add_argument(
        '--file3', type=str, default='./similarity/final/SimCOU_GOTF_Zhang et al._normal.csv')  # ANN
    parser.add_argument(
        '--file2', type=str, default='./similarity/final/Literature_similarity_topics=85_normal.csv')  # LIT
    parser.add_argument(
        '--file1', type=str, default='./similarity/final/DIN_similarity_normal.csv')  # DIN
    parser.add_argument('--targets', type=str,
                        default='./similarity/final/mutlimodal_fusion_sim_normal.csv') # multimodal-information
    args = parser.parse_args()

    # Fusion the similarity based on multimodal-information
    fusion_similarity(args.file2, args.file3, args.targets)
    fusion_similarity(args.targets, args.file4, args.targets)
    fusion_similarity(args.targets, args.file5, args.targets)

    time_end1 = time.time()
    print('total time (s) for multimodal-information fusion: ', time_end1 - time_start)

    # Fusion of disease similarity features based on DIN and multimodal-information
    dim = 1000
    target1 = './runTime/DIN+multimodal_sim_normal' + str(dim) + '.csv'
    MCBP_fusion(args.file1, args.targets, target1, dim)

    time_end2 = time.time()
    print('total time (s): ', time_end2 - time_start)

