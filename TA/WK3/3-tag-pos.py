# -*- coding: utf-8 -*-
"""
Updated on Fri Jun 26 2020
Workshop: IE - POS Tagging
@author: issfz
"""

import nltk
from nltk import word_tokenize, pos_tag

# ===== POS Tagging using NLTK =====

sent = '''Professor Tan Eng Chye, NUS Deputy President and Provost, and Professor 
Menahem Ben-Sasson, President of HUJ signed the joint degree agreement at NUS, 
in the presence of Ambassador of Israel to Singapore Her Excellency Amira Arnon 
and about 30 invited guests, on July 03, 2013.
'''

# The input for POS tagger needs to be tokenized first.
# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
sent_pos = pos_tag(word_tokenize(sent))
sent_pos

# A more simplified tagset - universal
#https://universaldependencies.org/u/pos/all.html
sent_pos2 = pos_tag(word_tokenize(sent), tagset='universal')
sent_pos2

# The wordnet lemmatizer works properly with the pos given
wnl = nltk.WordNetLemmatizer()
wnl.lemmatize('born', pos = 'v')

def lemmaN(wpos):
    nouns = [ (w, pos) for (w, pos) in wpos if pos in ('NN', 'NNS')]
    lemman = [ wnl.lemmatize(w, pos = 'n') if pos == 'NNS' else w for (w, pos) in nouns  ]
    return lemman

def lemmaV(wpos):
    verbs = [ (w, pos) for (w, pos) in wpos if pos.startswith('V') ]
    lemmav = [ wnl.lemmatize(w, pos = 'v') if pos != 'VB' else w for (w, pos) in verbs  ]
    return lemmav

lemmaN(sent_pos)
lemmaV(sent_pos)

#------------------------------------------------------------------------
# Exercise: remember the wordcloud we created last week? Now try creating 
# a wordcloud with only nouns, verbs, adjectives, and adverbs, with nouns 
# and verbs lemmatized.
#-------------------------------------------------------------------------