import numpy as np
import pandas as pd

reports = pd.read_table('osha.txt', header=None, names=["ID", "Title", "Content"])

reports['Text'] = reports['Title'] + '. ' + reports['Content']


from nltk.corpus import wordnet

synoset = wordnet.synsets("worker")

workers_list = []

for syno in synoset:
    for lemma in syno.lemmas():
        workers_list.append(lemma.name())
        if lemma.antonyms():
            workers_list.append(lemma.antonyms()[0].name())

pass