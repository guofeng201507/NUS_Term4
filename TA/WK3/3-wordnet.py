# -*- coding: utf-8 -*-
"""
Updated on Fri Jun 26 2020
Workshop: IE - WordNet as lexical resource
@author: issfz
"""
from nltk.corpus import wordnet as wn

# To look up a word
wn.synsets('book')
print (wn.synsets('book'))
# Look up with specified POS - NOUN, VERB, ADJ, ADV
print (wn.synsets('book', pos = wn.VERB))

# Let's examine a synset in more details: its definition, examples, lemma
ss = wn.synsets('book', pos = wn.VERB)[1]
print(ss.definition())
print(ss.examples())

ss.lemmas()
lem = ss.lemmas()[0]
lem.name()
ss.lemma_names()

# A synset has related synsets, for example, its hypernyms, hyponyms
ss = wn.synsets('vehicle')[0]
print (ss)
ss.hypernyms()
ss.hyponyms()

ss.hyponyms()[3].lemma_names()
# get "vehicle"'s hyponyms'closure words'lemma_names 
hyps = list(set(
                [w for s in ss.closure(lambda s:s.hyponyms())
                        for w in s.lemma_names()]))
print(hyps)
