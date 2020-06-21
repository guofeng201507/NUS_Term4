# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:16:43 2019

@author: isswan
"""

# In this workshop we perform topic modeling using gensim
# The "Class" labels here are only used for sanity check of the topics discovered later.
# Remember, in actual use of topic modelling, the documents DON'T come with labeled classes.
# It's unsupervised learning.

import numpy as np
import pandas as pd
news=pd.read_table('r8-train-all-terms.txt',header=None,names = ["Class", "Text"])
news.groupby('Class').size()

subnews=news[(news.Class=="trade")| (news.Class=='crude')|(news.Class=='money-fx') ]
subnews.head()
subnews.groupby('Class').size()


#########################################################Preprocessing
import nltk
from nltk.corpus import stopwords
mystopwords=stopwords.words("English") + ['one', 'become', 'get', 'make', 'take']
WNlemma = nltk.WordNetLemmatizer()

def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[ WNlemma.lemmatize(t.lower()) for t in tokens]
    tokens=[ t for t in tokens if t not in mystopwords]
    tokens = [ t for t in tokens if len(t) >= 3 ]
    return(tokens)


text = subnews['Text']
correct_labels = subnews['Class']
toks = text.apply(pre_process)

# Use dictionary (built from corpus) to prepare a DTM (using frequency)
import logging
import gensim 
from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Filter off any words with document frequency less than 3, or appearing in more than 80% documents
dictionary = corpora.Dictionary(toks)
print(dictionary)
dictionary.filter_extremes(no_below=3, no_above=0.8)
print(dictionary)

#dtm here is a list of lists, which is exactly a matrix
dtm = [dictionary.doc2bow(d) for d in toks]

lda = gensim.models.ldamodel.LdaModel(dtm, num_topics = 3,id2word = dictionary)

lda.show_topics(20)

##Note that different runs result in different but simillar results 
##Label the topics based on representing "topic_words"

dict = {0: 'crude', 1: 'trade', 2: 'money-fx'}

# Get the topic distribution of documents
doc_topics = lda.get_document_topics(dtm)
#show the topic distributions for the first 5 docs, 
for i in range(0, 5):
    print(doc_topics[i])

#Select the best topic (with highest score) for each document
from operator import itemgetter
top_topic = [ max(t, key=itemgetter(1))[0] for t in doc_topics ]
topics_perDoc = [ dict[t] for t in top_topic ]


####################################### How many dos in each topic?
labels, counts = np.unique(topics_perDoc, return_counts=True)
print (labels)
print (counts)

subnews.groupby('Class').size()


###############################################Evaluation 
# Now let's see how well these topics match the actual categories
##### Remember, in actual use of LDA, the documents DON'T come with labeled topics.
##### So nomally we can not access the confusion matrix unless we label some data manually 
import numpy as np
from sklearn import metrics
print(metrics.confusion_matrix(topics_perDoc, correct_labels))
print(np.mean(topics_perDoc == correct_labels) )
print(metrics.classification_report(topics_perDoc, correct_labels))

###########################Save and load pre-trained model
from gensim.test.utils import datapath
# Save model to disk.
temp_file = datapath("LDA_model")
lda.save(temp_file)
# Load a potentially pretrained model from disk.
lda = gensim.models.ldamodel.LdaModel.load(temp_file)
########################Query, the model using new, unseen documents
other_texts = [
    ['US', 'dollor', 'Yen','market','devalue','value'],#doc 1
    ['ship', 'import','export'],#doc 2
    ['petro', 'shell', 'esso']#doc 3
]

new_dtm = [dictionary.doc2bow(d) for d in other_texts]

for i in range(0, 3):
    unseen_doc = new_dtm[i]
    print(lda[unseen_doc])

dict = {0: 'crude', 1: 'trade', 2: 'money-fx'}

########################Update the model by incrementally training on the new corpus
lda.update(new_dtm)

for i in range(0, 3):
    unseen_doc = new_dtm[i]
    print(lda[unseen_doc])

