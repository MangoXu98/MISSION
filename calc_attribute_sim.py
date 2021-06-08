'''
Calculate disease similarity based on disease attributes
'''
import gensim
import gensim.models
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import *
import time


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

def process_similarity(pairs, word_vectors, part):
    '''
    Calculate the cosine similarity of disease pairs
    '''
    disease1, disease2, similarity = [], [], []
    for pair in tqdm(pairs):
        disease1.append(pair[0])
        disease2.append(pair[1])
        sim = cos_sim(word_vectors.get_vector(pair[0]), word_vectors.get_vector(pair[1]))
        similarity.append(sim)

    temp = pd.DataFrame(
        {'disease1': disease1, 'disease2': disease2, 'similarity': similarity})
    return temp


if __name__ == '__main__':
    time_start = time.time()
    print("=============== Load data... ================")
    df = pd.read_csv('./dataset/attributes/DO_metaData.csv', header=None)
    df_doid = pd.read_csv('./dataset/associations/doid_names_1754.csv')
    index2id = {}
    sentences = []

    for index, row in df_doid.iterrows():
        index2id[index] = row[0]
    for index, row in df.iterrows():
        string = str(row[0])+' '+str(row[1])+' '+str(row[2])
        sentences.append(string.split(' '))

    # Retrain the pre-trained model based on the attribute corpus
    print("=============== Begin training... ================")
    model = gensim.models.Word2Vec.load('./model/PMC_model/PMC_model.txt')
    model.min_count = 1
    model.build_vocab(sentences, update=True)
    model.train(sentences, total_examples=model.corpus_count, epochs=100)
    word_vectors = model.wv
    print('The training of model has been completed...')

    time_end1 = time.time()
    print("\r\n Total time (s) of training:", time_end1 - time_start)

    dict = {}
    for index in index2id.keys():
        dict[index2id[index]] = word_vectors.get_vector(index2id[index])

    pairs = list(combinations(list(dict.keys()), 2))

    chunks = partition(pairs, 1)
    futures = {}

    df_res = pd.DataFrame({'disease1': [], 'disease2': [], 'similarity': []})


    print("=============== Begin parallel computing... ================")
    with ProcessPoolExecutor(max_workers=1) as executor:
        part = 1
        for c in chunks:
            job = executor.submit(process_similarity, c, word_vectors, part)
            futures[job] = part
            part += 1
        for job in as_completed(futures):
            temp = job.result()
            df_res = pd.concat([df_res, temp])
    
    df_res.to_csv('./runTime/att_similarity.csv', index=None)

    time_end2 = time.time()
    print('total time (s): ', time_end2 - time_start)
