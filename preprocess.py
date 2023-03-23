# -*- coding: utf-8 -*-

import logging
import sys
import os.path
import requests
import json
from tqdm import tqdm


def get_instance_concept(file):
    ent_concept = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            cpt = line[0]
            ent = line[1]
            if ent not in ent_concept:
                ent_concept[ent] = []
            ent_concept[ent].append(cpt)

    return ent_concept


if __name__ == '__main__':
    import spacy
    file = '/Downloads/data-concept/data-concept-instance-relations.txt'
    k = 3
    nlp = spacy.load("en_core_web_sm")
    ent_concept = get_instance_concept(file)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}
    f_w = open('news2.tsv', 'w', encoding='utf-8')
    text = 'None'
    with open('news.txt', 'r', encoding='utf-8') as f:
        for ii, line in enumerate(f):
            text = line.lower().replace('\n','')

            doc = nlp(text)
            obj = []

            for ent in doc.ents:
                obj.append(ent.text)
                print(ent.text, ent.start_char, ent.end_char, ent.label_)
            concept = []
            if len(obj) == 0:
                concept.append('None')

            for ent in obj:
                # concept_sen = "https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance="+i+"&topK=3"
                # response = requests.get(concept_sen, headers=headers)

                # content = json.loads(response.content)
                if ent in ent_concept:
                    length = len(ent_concept[ent])
                    length = k if length > k else length
                    concept.extend(ent_concept[ent][0:length])
                else:
                    concept.append(ent)
            f_w.write(text + '\t' + ' '.join(concept) + '\t' + '\n')

                # if len(content) > 0:
                #     # print(content.keys())
                #     concept.extend(content.keys())
                # else:
                #     concept.append(i)
                # time.sleep(0.5)
            # print(concept)
            # print(' '.join(concept))

    f_w.close()
