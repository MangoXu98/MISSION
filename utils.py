import numpy as np
import json
import os
import pandas
import pickle
import csv

def reverse_map(m):
    '''
    :param m: row dictionary {k:v}
    :return: reversed dictionary {v:k}
    '''
    return {v: k for k, v in m.items()}

def partition(lst, n):
    '''
    split a list to n partition
    :param lst: the source list
    :param n: the number of n
    :return: a list contain n parition [[],[],...,[]]
    '''
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = cos
    return sim

    
def array_normalization(x):
    Max = max(x)
    Min = min(x)
    return [(float(i)-Min)/(Max-Min) for i in x]


def load_json_file(json_file):
    """
        Load json file from local.

    :param json_file:
    :return:
    """
    # json_file：json/F1367.json'
    with open(json_file, 'r') as load_f:
        json_content = json.load(load_f)
        print(json_content)

    return json_content


def save_to_json_file(content, fiori_id):
    """
        Save content in a json file.

    :param content:
    :param fiori_id:
    :return:
    """

    filename = "../json/" + str(fiori_id) + '.json'

    if os.path.exists(filename):  # 若json文件存在，则直接返回文件名称
        print(filename, "JSON file exists. Skipping storing.")
        return filename

    # 方式1
    # with io.open(filename, "wb", encoding="utf-8") as f:
    #     f.write(content)

    # 方式2
    with open(filename, "w") as dump_f:
        json.dump(content, dump_f)

    print("JSON file saved.")

    return filename  # json file name


def open_csv(filename):
    """
        Open csv file and return data.

    :param filename:
    :return: data in the csv file.
    """
    data = pandas.read_csv(filename, encoding='windows-1252')
    print("Opening csv file successfully! [", filename, "]")
    return data


def is_pkl_exist(filename):
    """
        Check if a pkl file exists.

    :param filename:
    :return:  True if the file exists.
    """
    if os.path.exists("pkl/" + filename + ".pkl"):
        print(filename, "exists!")
        return True
    else:
        return False


def save_on_disk(data, filename):
    """
        Save the pkl file on disk.
    """
    filename = "pkl/new/" + filename + ".pkl"
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle)
    print("Data-->", filename, " has been saved in pkl file on disk.")

    return True


def load_pkl_from_disk(filename):
    """
        Load pkl file from disk, return data.

    :param filename:
    :return:
    """
    fr = open("pkl/" + filename + '.pkl', 'rb')  # open的参数是pkl文件的路径
    data = pickle.load(fr)  # 读取pkl文件的内容
    # data = bytes(data, encoding="utf-8")
    fr.close()  # 关闭文件
    print("Load pkl file [", filename, ".pkl]")
    return data


def save_all_to_csv(attr_name_list, all_app_attr_val_dict, filename):
    """
         Save all data into csv file.

    :param attr_name_list:
    :param all_app_attr_val_dict:
    :param filename:
    :return:
    """
    filename = "../save/" + filename + ".csv"

    with open(filename, "w", newline='') as myfile:  # w，若为wb则是byte形式

        print(type(attr_name_list))
        print(all_app_attr_val_dict)

        attr_name_list.insert(0, 'Fiori_id')  # 字段名称第一个插入 Fiori_id

        dict_writer = csv.DictWriter(myfile, fieldnames=attr_name_list)  # 字段名称为 query name
        dict_writer.writeheader()   # 首行写入属性字段名称

        for app_id, app_dict in all_app_attr_val_dict.items():
            dict_writer.writerow(app_dict)  # 写入数据，一次写入一个字典到一行(一个app的内容）

    print("Data has been saved to csv.")

    return True


def save_dict_to_csv(mydic, attr_name_list, filename):
    """
            doc_topic_prob_vec_dict = {
                doc_id: [prob_1, ..., prob_n]
                ...
            }
    """
    filename = "save/" + filename + ".csv"

    with open(filename, "w", newline='') as myfile:

        writer = csv.writer(myfile)

        writer.writerow(attr_name_list)

        for doc_id, topic_prob in mydic.items():
            writer.writerow(topic_prob)

    print("Data has been saved to csv.")

    return True


''' Test '''
def save_csv_test(attr_list_xml, data, filename):
    filename = "../save/" + filename + ".csv"

    with open(filename, "w", newline='') as myfile:  # w，若为wb则是byte形式

        # writer.writerow(['Column1', 'Column2', 'Column3'])

        # print(type(attr_list_xml))   # <class 'list'>
        print("len of dict", len(data))

        attr_list_xml.insert(0, 'F_id')  # 字段名称第一个插入F_id

        dict_writer = csv.DictWriter(myfile, fieldnames=attr_list_xml)
        dict_writer.writeheader()

        for app_id, app_dict in data.items():
            dict_writer.writerow(app_dict)  # 写入数据，一次写入一个字典(一个app的内容）

    print("Data has been saved.")

    return


def save_wrong_release_app(wrong_release_id_set, filename):
    """
        Save to txt.
    :param wrong_release_id_set:
    :param filename:
    :return:
    """
    print("len of wrong id:", len(wrong_release_id_set))
    print(wrong_release_id_set)

    write_to_txt(wrong_release_id_set, filename)
    print("Apps with wrong release_id saved.")
    return


def write_to_txt(data, filename):
    """
        Save data to txt file.

    :param data:
    :param filename:
    :return: filename (with path and format suffix)
    """
    filename = "save/" + filename + ".txt"

    f = open(filename, 'w', encoding="utf-8")

    print("Write data to txt file. Length of data:", len(data))

    for data_element in data:
        f.writelines(data_element + '\n')

    # f.writelines([line + '\n' for line in data])

    f.close()

    print("Write data to txt finished.")

    return filename


def write_dict_to_txt(dic, filename):
    """
           Save data to txt file.

       :param data:
       :param filename:
       :return: filename (with path and format suffix)
    """

    filename = "save/" + filename + ".txt"

    f = open(filename, 'w', encoding="utf-8")

    print("Write dic to txt file. Length of data:", len(dic))

    for key, val in dic.items():
        f.writelines(str(key) + "##" + val + '\n')  # line: mesh_id ## text

    # f.writelines([line + '\n' for line in data])

    f.close()

    print("Write dic to txt finished.")

    return filename


def load_relation_dict():
    with open("./dict/d2g.json", "r") as f:
        d2g = json.load(f)
    with open("./dict/d2c.json", "r") as f:
        d2c = json.load(f)
    with open("./dict/d2h.json", "r") as f:
        d2h = json.load(f)
    with open("./dict/g2d.json", "r") as f:
        g2d = json.load(f)
    with open("./dict/c2d.json", "r") as f:
        c2d = json.load(f)
    with open("./dict/h2d.json", "r") as f:
        h2d = json.load(f)

    return {'D2G':d2g,'D2H':d2h,'D2C':d2c,'C2D':c2d,'G2D':g2d,'H2D':h2d}
