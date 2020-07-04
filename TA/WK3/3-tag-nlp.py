# -*- coding: utf-8 -*-
"""
Updated on Fri Jun 26 2020
Workshop: IE - Using NLP software
@author: issfz
"""

#We can use CoreNLP from Stanford NLP group to perform the NLP tasks.
# ===== Java is required =====
''' 
   Java installation if not installed yet (>1.8)
   Download and install java from https://www.java.com/en/download/

'''

# Set JAVAHOME variable to the directory containin Java on your computer
import os
java_path = 'C:\\Program Files\\Java\\jre1.8.0_131\\bin\\java.exe'
os.environ['JAVAHOME'] = java_path

       
# ===== Install Standford CoreNLP =====
'''
# 1. Download the latest version
#   from http://nlp.stanford.edu/software/stanford-corenlp-latest.zip
# 2. Unzip to your target folder, eg. D:/tools/
'''

from nltk.parse import CoreNLPParser, CoreNLPDependencyParser
from nltk import word_tokenize

'''
# 3. Go to the CoreNLP installation folder
#    Start CoreNLP server first at the windows commandline:
#    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000    
# 4. Open web browser, go to http://localhost:9000
     Test CoreNLP on the web UI first.
'''


pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')

sent = '''Professor Tan Eng Chye, NUS Deputy President and Provost, and Professor 
Menahem Ben-Sasson, President of HUJ signed the joint degree agreement at NUS, 
in the presence of Ambassador of Israel to Singapore Her Excellency Amira Arnon 
and about 30 invited guests, on July 03, 2013.
'''

print('Tokens:', list(pos_tagger.tokenize(sent)))
print('Part of Speech tags:', list(pos_tagger.tag(word_tokenize(sent))))
print('Part of Speech tags:', list(pos_tagger.tag(sent.split())))


ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
ner_output = list(ner_tagger.tag(sent.split()))
ner_output

# To get the entities from the tagged result
from itertools import groupby

for tag, chunk in groupby(ner_output, lambda x:x[1]):
    if tag != "O":
        print("%-12s"%tag, " ".join(w for w, t in chunk))



parser = CoreNLPParser(url='http://localhost:9000')
sent2 = "The epidemic has caused severe damage."
print('Tokens:', list(parser.tokenize(sent2)))
print('Constituency Parsing:', list(parser.parse(sent2.split())))

(output, ) = parser.parse(sent2.split())
output.pretty_print()

#you can also use raw_parse(), which takes in a sentence as string
print('Constituency Parsing:', list(parser.raw_parse(sent2)))


parserD = CoreNLPDependencyParser(url='http://localhost:9000')

outputD,  = parserD.raw_parse(sent2)
print(outputD.to_conll(4))
for governor, dep, dependent in outputD.triples():
    print(governor, dep, dependent)


'''
Alternative way: using spaCY!

Installation of spaCY and the required models:
    pip install -U spacy
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_md
'''

import spacy

#load the required model
nlp = spacy.load("en_core_web_sm")
nlp.pipeline


#process a sentence
doc1 = nlp(sent)

#detailed results
for token in doc1:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.head,
            token.shape_, token.is_alpha, token.is_stop)

for ent in doc1.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    
# Disable the pipeline components that you don't need.
# Pprocess multiple sentences more efficiently using pipe()
sents = [ '''The National University of Singapore and The Hebrew University of 
         Jerusalem (HUJ) are launching a Joint Doctor of Philosophy (PhD) 
         degree programme in biomedical science from August 2013.''', 
         '''Professor Tan Eng Chye, NUS Deputy President (Academic Affairs) and
         Provost, and Professor Menahem Ben-Sasson, President of HUJ signed the
         joint degree agreement at NUS, in the presence of Ambassador of Israel
         to Singapore Her Excellency Amira Arnon and about 30 invited guests.''',
         '''Students enrolled in the programme will divide their time between 
         both campuses in Singapore and Jerusalem, Israel, spending a minimum 
         of nine months at each institution. ''',
         '''Two NUS students have already been selected for the inaugural intake
         and they will begin their programme in the new academic year starting 
         this August.''']


docs = nlp.pipe(sents, disable=["tagger", "parser"])

for doc in docs:
    for ent in doc.ents:
        print(ent.text, ent.label_)
    print('\n')
