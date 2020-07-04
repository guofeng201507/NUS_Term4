# -*- coding: utf-8 -*-
"""
Updated on Fri Jun 26 2020
Workshop: IE - Named Entity Recognition
@author: issfz
"""

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# ===== POS Tagging and NER using NLTK =====

sent = '''Professor Tan Eng Chye, NUS Deputy President and Provost, and Professor 
Menahem Ben-Sasson, President of HUJ signed the joint degree agreement at NUS, 
in the presence of Ambassador of Israel to Singapore Her Excellency Amira Arnon 
and about 30 invited guests, on July 03, 2013.
'''

# The input for POS tagger needs to be tokenized first.
sent_pos = pos_tag(word_tokenize(sent))
sent_pos

# ===== NER using NLTK =====
# The input for the NE chunker needs to have POS tags.
sent_chunk = ne_chunk(sent_pos)
print(sent_chunk)

# ===== Now try creating your own named entity and noun phrase chunker ====
# We need to define the tag patterns to capture the target phrases and use 
# RegexParser to chunk the input with those patterns.
# Some minimal tag patterns are given here. 

grammar = r"""
  NE: {<NNP> of<IN> <NNP>}      # chunk NE sequences of proper nouns
  NP: {<DT><NN>}   # chunk noun phrase by DT+NN  
"""


cp = nltk.RegexpParser(grammar)

result = cp.parse(pos_tag(word_tokenize("Donald Trump is a crazy leader.")))
print(result)
result.draw()

result = cp.parse(sent_pos)
print(result)



#------------------------------------------------------------------------
# Exercise: modify the above tag patterns to capture the NEs and NPs in the 
# example sentence. 
#-------------------------------------------------------------------------