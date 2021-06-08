"""
Calculate the disease similarity based on literature
"""
import os
import string
import pandas
from time import time
from terms import *
from utils import *
import xml.dom.minidom as xmldom
import xml.etree.ElementTree as ET
from itertools import *
from itertools import combinations
from gensim.test.utils import common_texts
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
import nltk


tokenizer = RegexpTokenizer(r'\w+')

en_stop = stopwords = nltk.corpus.stopwords.words(
    "english")   # create English stop words list

p_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


print("len of en_stop", len(en_stop))
supp_stop_list = []
for i in string.digits:
    en_stop.append(i)
for e in string.ascii_letters:
    en_stop.append(e)
en_stop = en_stop + ['data', 'ed', 'national',
                     'new', 'may', 'type', 'use', 'used', 'work']


def get_corpus_from_dict(mesh_merged_dict):
    print("Fetching raw txt data from mesh_merged_dict...")

    docs = []

    for mesh_id, merged_text in mesh_merged_dict.items():
        docs.append(merged_text)

    print("# of texts in raw docs:", len(docs))

    return docs


def get_doId(doid_file):
    """
        Get DO id from file.
    """
    doid_dict = defaultdict(list)  # key: mesh_id, val: none
    data = open_csv(doid_file)
    rows_num = len(data)

    if rows_num == 0:
        print("File is empty!")
        return False

    print("total rows of [", doid_file, "]:", rows_num)

    column1 = data.columns[0]  # col 1, mesh_id
    for row in range(len(data)):
        # id1, remove the blanks in the start and end of the string
        mesh_id = data[column1][row].strip()
        meshid_dict[mesh_id] = {}  # key: mesh_id, val: none

    print("Get all ids, total num:", len(doid_dict))
    return doid_dict


def preprocessing_doc(docs, do_dict):
    """
        Tokenize, remove stop words, stemming.
    """
    print("======================= Pre-processing corpus... ========================")
    key_list = list(do_dict.keys())  # convert the keys into list
    print("Pre-processing raw docs...\r\n len of do_dict is:", len(key_list))

    clean_docs = {}

    # list for tokenized documents in loop
    clean_texts = []

    for i, doc_a in enumerate(docs):

        # clean and tokenize document string
        raw = doc_a.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [j for j in tokens if not j in en_stop]

        ''' use the WordNet lemmatizer from NLTK. 
            A lemmatizer is preferred over a stemmer in this case because it produces more readable words.  '''
        lemmalized_tokens = [lemmatizer.lemmatize(
            token) for token in stopped_tokens]

        clean_docs[key_list[i]] = lemmalized_tokens

        # add tokens to list
        clean_texts.append(lemmalized_tokens)

    return clean_texts


def trainLDA(clean_docs, ldamodel_name_prefix, lda_num_topics):
    """
        Train LDA model.
    """
    print("=============== Training LDA model... topic_num=",
          lda_num_topics, " ================")

    lda_model_filename = "model/LDA/" + ldamodel_name_prefix + ".model"

    if os.path.exists(lda_model_filename):
        print(lda_model_filename, "exists!")
        corpus = load_pkl_from_disk(ldamodel_name_prefix + "_corpus")
        dic = load_pkl_from_disk(ldamodel_name_prefix + "_dictionary")
        ldamodel = models.ldamodel.LdaModel.load(lda_model_filename)
        return corpus, dic, ldamodel

    ''' turn our tokenized documents into an (id -> term) dictionary. '''
    dictionary = corpora.Dictionary(clean_docs)

    ''' convert tokenized documents into a document-term matrix '''
    corpus = [dictionary.doc2bow(text) for text in clean_docs]

    ldamodel = models.ldamodel.LdaModel(
        corpus=corpus, num_topics=lda_num_topics, id2word=dictionary, passes=10)

    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    ''' Save model to disk. '''
    ldamodel.save(lda_model_filename)
    print("LDA model saved.")

    save_on_disk(corpus, ldamodel_name_prefix + "_corpus")
    save_on_disk(dictionary, ldamodel_name_prefix + "_dictionary")

    return corpus, dictionary, ldamodel


def get_new_doc_topic_distribution(corpus_dictionary, ldamodel, new_docs):
    """
        Get topic vector for new docs by the trained LDA model.

        Topic vector: a list of tuples, eg. [(topic_id, prob)]
    """
    print("Getting new doc topic distribution...")

    doc_topic_full_vec_dict = {}  # document with a topic vector
    doc_topic_prob_vec_dict = {}  # no topic id

    for doc_id, text in new_docs.items():
        # convert the text to bag of words
        bow = corpus_dictionary.doc2bow(text.lower().split())
        topic_vec = ldamodel.get_document_topics(
            bow, minimum_probability=0.0)  # get the topic vector of each doc

        ''' save the topic id and probability at the same time (topic_id, probability) '''
        doc_topic_full_vec_dict[doc_id] = topic_vec

        ''' only save topic probability  '''
        topic_prob_list = []  # a list of prob
        for (topicID, prob) in topic_vec:
            topic_prob_list.append(prob)

        # only prob, no topic id
        doc_topic_prob_vec_dict[doc_id] = topic_prob_list

    return doc_topic_full_vec_dict, doc_topic_prob_vec_dict


def get_pub_dict(filename):
    print("=============== Load disease document... ================")
    folder_path = 'dataset/text/'
    suffix = '.txt'
    pub_dict = {}
    termId_dict = defaultdict(list)  # key: do_id, val: none
    data_all = pandas.read_csv(filename, encoding='utf-8')
    data = data_all
    rows_num = len(data)

    if rows_num == 0:
        print("File is empty!")

    column0 = data.columns[0]  # col 1, term_id
    column1 = data.columns[1]  # col 2, term_name

    for row in range(len(data)):
        # id1, remove the blanks in the start and end of the string
        term_id = str(data[column0][row]).strip()
        term_title = str(data[column1][row]).strip()
        target = folder_path + term_title + suffix

        termId_dict[term_id] = {}  # key: do_id, val: none

        with open(target, encoding='utf-8') as txtreader:
            content = txtreader.read()
            pub_dict[term_id] = term_title + ',' + content

    return termId_dict, pub_dict


def cal_dpair_sim(doc_topic_prob_vec_dict):
    """
        Calculate the similarity of disease pairs based on LDA feature vectors.
    """
    print("=============== Calculate disease similarity... ================")
    doc_sim_dict = {}
    doc_id_set = doc_topic_prob_vec_dict.keys()
    doc_pair_list = list(combinations(doc_id_set, 2))  # all doc pairs

    for doc_pair in doc_pair_list:
        doc_1_topic_prob = doc_topic_prob_vec_dict[doc_pair[0]]
        doc_2_topic_prob = doc_topic_prob_vec_dict[doc_pair[1]]

        doc_sim_dict[doc_pair] = cal_cosine_sim(
            doc_1_topic_prob, doc_2_topic_prob)

    print("len of doc_sim_dict by emb:", len(doc_sim_dict))
    return doc_sim_dict


def cal_cosine_sim(vec1, vec2):
    '''
        Cosine similarity
    '''
    multiple = 0
    norm_vec1 = 0
    norm_vec2 = 0
    for v1, v2 in zip(vec1, vec2):
        multiple += v1 * v2
        norm_vec1 += v1 ** 2
        norm_vec2 += v2 ** 2
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    else:
        return multiple / ((norm_vec1 * norm_vec2) ** 0.5)


if __name__ == '__main__':
    time_start = time.time()
    ''' 1. Merge term info and save the result on disk (txt file). '''
    filename = "dataset/associations/doid_names_1754.csv"
    termId_dict, pubmed_processed_dict = get_pub_dict(filename)

    raw_docs = get_corpus_from_dict(
        pubmed_processed_dict)  # a list of disease texts

    ''' 2. Preprocess the raw texts from corpus'''
    clean_docs_list = preprocessing_doc(raw_docs, termId_dict)

    time_end2 = time.time()

    ''' 3. train LDA model by clean corpus'''
    lda_num_topics = 85
    ldamodel_name_prefix = "lda_pubmed_from_do_numTopics=" + \
        str(lda_num_topics)
    corpus, dictionary, ldamodel = trainLDA(
        clean_docs_list, ldamodel_name_prefix, lda_num_topics)

    ''' 4. Get vectors of new docs by trained model'''
    doc_topic_full_vec_dict, doc_topic_prob_vec_dict = get_new_doc_topic_distribution(
        dictionary, ldamodel, pubmed_processed_dict)

    time_end1 = time.time()
    print("\r\n Total time (s) for training:", time_end1 - time_end2)

    ''' 5. Calculate the similarity between disease pairs. '''
    dpair_sim = cal_dpair_sim(doc_topic_prob_vec_dict)
    save_on_disk(dpair_sim, "LDA_pubmed_sim_" + str(lda_num_topics))
    
    time_end = time.time()
    print("\r\n Total time (s):", time_end - time_start)
