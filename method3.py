# -*- coding: utf-8 -*-
# 这一段是实体概念扩展
import logging
import sys
import subprocess
import os

from sentence_transformers import SentenceTransformer, util


def distance(sentence1):
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    # compute similarity scores of two embeddings

    cosine_negative = util.pytorch_cos_sim(embedding1, embedding_negative)
    cosine_positive = util.pytorch_cos_sim(embedding1, embedding_positive)

    max_list = {}
    max_list["negative"] = cosine_negative.item()
    max_list["positive"] = cosine_positive.item()

    sorted = dict_sort(max_list)
    return list(sorted.items())[0]


# program = os.path.basename(sys.argv[0])
# logger = logging.getLogger(program)
# logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')


#####此处增加################################################################################################################################
# agnews
model = SentenceTransformer('stsb-roberta-large')
label_dir = ["negative", "positive"]
limit = 0.37
embedding_negative = model.encode("negative", convert_to_tensor=True)
embedding_positive = model.encode("positive", convert_to_tensor=True)
###########################################################################################################################################

def dict_sort(d):
    x = 50  #Verbalizer limit
    d.items()
    L = list(d.items())
    L.sort(key=lambda x: x[1], reverse=True)
    return dict(L[0:x])


def get_instance_concept(file):
    ent_concept = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            cpt = line[0]
            ent = line[1]
            if ent not in ent_concept:
                ent_concept[ent] = []  # {"apple":"inc"}
            ent_concept[ent].append(cpt)

    return ent_concept


def bert_cluster():
    start1 = time.perf_counter()
    word = []
    # f_co = open('./ag_snippets_6.tsv', 'r', encoding='utf-8')
    for line3 in raw_word:
        line3 = line3.replace("\n", "")
        line3 = line3.split("\t")
        word.append(line3[1].split(" "))
    # f_co.close()
    #################此处要对dict1插入以及 sort  20220408 20220408 20220408 20220408 以及写入 verbalizer
    for items in word:
        for item in items:
            score_list = distance(item)
            # print(item)
            if score_list[1] <= limit:
                continue
            else:
                if score_list[0] == "negative":
                    dict1[0][item] = score_list[1]
                else:
                    dict1[1][item] = score_list[1]
                ######一开一关
    end1 = time.perf_counter()
    ####此处进行dict1的排序 并只保存前 50个
    dict1[0] = dict_sort(dict1[0])
    dict1[1] = dict_sort(dict1[1])
    print("#######bert结束#####")
    print('Running111 time cluster: %s Seconds' % (end1 - start1))


if __name__ == '__main__':
    import time
    from rake_nltk import Rake

    start = time.perf_counter()
    file = './data-concept-instance-relations.txt'
    k = 3  ########此处修改最大概念数
    ent_concept = get_instance_concept(file)
    text = 'None'
    r = Rake()

    # 初始化
    ###几个概念词
    ###原 expand_ex.txt four_label_sni.txt
    ###原始句子
    raw_word = []
    ###全局标签词典
    dict1 = []
    dict1.append({'negative': 1})
    dict1.append({'positive': 1})

    count = 0
    with open(f'./datasets/TextClassification/DL-Clickbait/train.csv', 'r', encoding='utf-8') as f:
        for ii, line in enumerate(f):
            line = line.split('\",\"')
            label_num = line[0].replace('\"', ' ')
            sentence = line[1].replace('\"', ' ')
            sentence = sentence.replace('(', '')
            sentence = sentence.replace(')', '')
            sentence = sentence.replace(',', '')

            line1 = sentence.split()
            text = ' '.join(line1[0:])
            label = label_dir[int(label_num) - 1]
            obj = []

            for ent in line1:
                obj.append(ent)
            concept = []
            if len(obj) == 0:
                concept.append('None')
            for ent in obj:
                if ent in ent_concept:
                    length = len(ent_concept[ent])
                    length = k if length > k else length
                    concept.extend(ent_concept[ent][0:length])
                else:
                    concept.append(ent)
            raw_word.append(text + '\t' + ' '.join(concept) + '\t' + label + '\n')

    bert_cluster()

    f_co_2 = open('./scripts/TextClassification/DL-Clickbait/manual_verbalizer.txt', 'w', encoding='utf-8')
    for item in dict1:
        sstr1 = ""
        for key, value in item.items():
            sstr1 = sstr1 + key + ","
        sstr1 = sstr1 + "\n"
        print(sstr1)
        f_co_2.write(sstr1)
    f_co_2.close()
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))