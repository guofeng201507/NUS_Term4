# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:13:28 2020

@author: isswan
"""
#pip install keras
#pip install tensorflow
#pip install plot_keras_history
#pip install seaborn



from keras.utils import np_utils
from keras.preprocessing import sequence

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda, Reshape

from keras.layers import Input
from keras.models import Model


from keras.layers import dot



from tensorflow.keras.activations import relu

from nltk import word_tokenize, sent_tokenize
from gensim.corpora.dictionary import Dictionary
import numpy as np

from keras.preprocessing.sequence import skipgrams
import gensim


AlotOftext = """Language users never choose words randomly, and language is essentially
non-random. Statistical hypothesis testing uses a null hypothesis, which
posits randomness. Hence, when we look at linguistic phenomena in corpora, 
the null hypothesis will never be true. Moreover, where there is enough
data, we shall (almost) always be able to establish that it is not true. In
corpus studies, we frequently do have enough data, so the fact that a relation 
between two phenomena is demonstrably non-random, does not support the inference 
that it is not arbitrary. We present experimental evidence
of how arbitrary associations between word frequencies and corpora are
systematically non-random. We review literature in which hypothesis testing 
has been used, and show how it has often led to unhelpful or misleading results.""".lower()



tokenized_text = [word_tokenize(sent) for sent in sent_tokenize(AlotOftext)]
vocab = Dictionary(tokenized_text)


dict(vocab.items())
vocab.token2id['corpora']
vocab[2]
sent0 = tokenized_text[0]
vocab.doc2idx(sent0)
vocab.add_documents([['PAD']])
dict(vocab.items())
vocab.token2id['PAD']

corpusByWordID = list()
for sent in  tokenized_text:
    corpusByWordID.append(vocab.doc2idx(sent))

vocab_size = len(vocab)
embed_size = 100
window_size = 2 # context window size

print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(vocab.items())[:10])


############################################

def generate_cbow_context_word_pairs(corpusByID, window_size, vocab_size):
    context_length = window_size*2
    X=[]
    Y=[]
    for sent in corpusByID:
        sentence_length = len(sent)
        for index, word in enumerate(sent):
            context_words = []
            label_word   = []            
            start = index - window_size
            end = index + window_size + 1
            
            context_words.append([sent[i] 
                                 for i in range(start, end) 
                                 if 0 <= i < sentence_length 
                                 and i != index])
            label_word.append(word)
            if start<0:
                x = sequence.pad_sequences(context_words, maxlen=context_length,padding='pre',value=vocab.token2id['PAD'])
                y = np_utils.to_categorical(label_word, vocab_size)
                X.append(x)
                Y.append(y)
                continue
            if end>=sentence_length:
                x = sequence.pad_sequences(context_words, maxlen=context_length,padding='post',value=vocab.token2id['PAD'])
                y = np_utils.to_categorical(label_word, vocab_size)
                X.append(x)
                Y.append(y)
                continue
            else:
                X.append(sequence.pad_sequences(context_words, maxlen=context_length))
                Y.append(y)
                continue
           
    return X,Y
            
# Test this out for some samples


X,Y = generate_cbow_context_word_pairs(corpusByWordID, window_size, vocab_size) 
   
for x, y in zip(X,Y):
    print('Context (X):', [vocab[w] for w in x[0]], '-> Target (Y):', vocab[np.argwhere(y[0])[0][0]])

 
############################################
#define and train the model
cbow = Sequential()
###hint:output_dim = the shape of embedding matrix
###hint:input_length = the length of training sample
cbow.add(Embedding(input_dim=vocab_size, output_dim=???, input_length=???))
cbow.add(Lambda(lambda x: relu(K.mean(x, axis=1)), output_shape=(embed_size,)))
###hint:the total numbser of possible label words
###hint:activation='softmax' or 'sigmoid'
cbow.add(Dense(???, activation='???'))
###hint:loss='categorical_crossentropy' or 'binary_crossentropy'
cbow.compile(loss='???', optimizer='sgd')
cbow.summary()


for epoch in range(1000):
    loss = 0.
    for x, y in zip(X,Y):
        loss += cbow.train_on_batch(x, y)
    print(epoch, loss)

##########################################
## Save the wordvectors
f = open('Cbow_vectors.txt' ,'w')
f.write('{} {}\n'.format(vocab_size-1, embed_size))
vectors = cbow.get_weights()[0]
for key in vocab:
    str_vec = ' '.join(map(str, list(vectors[key, :])))
    f.write('{} {}\n'.format(vocab[key], str_vec))
f.close()
##########################################
## Load the vectors back and validate
w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)

w2v.most_similar(positive=['that'])
w2v.most_similar(negative=['that'])
#############################################################
#############################################################
#Skipgram

# generate skip-grams with both positive and negative examples
skip_grams = [skipgrams(sent, vocabulary_size=vocab_size, window_size=2) for sent in corpusByWordID]

# view sample skip-grams
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(10):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
        vocab[pairs[i][0]], pairs[i][0],           
        vocab[pairs[i][1]], pairs[i][1], 
        labels[i]))

#########################################################
#define the model
input_word = Input((1,))
input_context_word = Input((1,))

word_embedding = Embedding(input_dim=vocab_size, output_dim=???,input_length=1,name='word_embedding')
context_embedding = Embedding(input_dim=vocab_size, output_dim=???,input_length=1,name='conotext_embedding')

word_embedding = word_embedding(input_word)
word_embedding_layer = Reshape((embed_size, 1))(word_embedding)

context_embedding = context_embedding(input_context_word)
context_embedding_layer = Reshape((embed_size, 1))(context_embedding)

# now perform the dot product operation word_embedding_vec * context_embedding_vec
dot_product = dot([???, ???], axes=1)
dot_product = Reshape((1,))(dot_product)

###hint:the total numbser of possible label words
###hint:activation='softmax' or 'sigmoid'
outputLayer = Dense(???, activation='???')(dot_product)

model = Model(input=[input_word, input_context_word], output=outputLayer)

###hint:loss='categorical_crossentropy' or 'binary_crossentropy'
model.compile(loss='???', optimizer='adam')

# view model summary
print(model.summary())
############################################################
#train the model

for epoch in range(1, 100):
    loss = 0
    for i, elem in enumerate(skip_grams):
        pair_first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        pair_second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [pair_first_elem, pair_second_elem]
        Y = labels
        if i % 10000 == 0:
            print('Processed {} (skip_first, skip_second, relevance) pairs'.format(i))
        loss += model.train_on_batch(X,Y)  

    print('Epoch:', epoch, 'Loss:', loss)

#########################################################
#get the embeding matrix
weights = model.get_weights()
## Save the wordvectors
f = open('skipgram_vectors.txt' ,'w')
f.write('{} {}\n'.format(vocab_size-1, embed_size))
vectors = model.get_weights()[0]
for key in vocab:
    str_vec = ' '.join(map(str, list(vectors[key, :])))
    f.write('{} {}\n'.format(vocab[key], str_vec))
f.close()
##########################################
## Load the vectors back and validate
w2v = gensim.models.KeyedVectors.load_word2vec_format('./skipgram_vectors.txt', binary=False)
w2v.most_similar(positive=['the'])
w2v.most_similar(negative=['the'])

################################################################################################
#Excerise: modeify the skipegram_model to share the same embeding layer between word and context
#Dicuss: which is better? Why?  






