# -*- coding: utf-8 -*-
"""
Workshop : Text preparation step by step
Author: Fan Zhenzhen
"""
import os
import json
import string
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords


#load data for NLTK (you just need to run it once only)
#nltk.download()
#at the 'NLTK Downloader' window, choose 'book', then click 'Download'

# We'll process just one article for illustration

# Set folder path to the directory where the files are located
# And open the file(s). For example:
#folder_path = 'D:\MTECH\TextMining\Day2'
#data =  open(os.path.join(folder_path, "Article_1.json"), "r")

# Or open the JSON data file from your current working directory
data =  open("Article_1.json", "r")

# Load in the JSON object in the file
jdata = json.load(data)
jdata.keys()

# Extract the URL and the Text from within the JSON object
url=jdata['URL']
url

text=jdata['Text']
text
type(text)
len(text)
text[:20]

# Conver the free text into tokens
tokens = word_tokenize(text)
type(tokens)

# A little exploration: How many words in this article? 
len(tokens)
tokens[:30]

#How many unique words?
unique = set(tokens)
len(unique)
len(tokens)/len(unique)
sorted(unique)[:30]

# Any single character words?
single=[w for w in unique if len(w) == 1 ]
len(single)
single

# Frequency distribution of the words
tokens.count('gluten')
fd = nltk.FreqDist(tokens)
fd.most_common(50)
fd.plot(50)

# How long are the words?
fd_wlen = nltk.FreqDist([len(w) for w in unique])
sorted(fd_wlen.items())

# What about bigrams and trigrams?
bigr = nltk.bigrams(tokens[:10])
trigr = nltk.trigrams(tokens[:10])
tokens[:10]
list(bigr)
list(trigr)

# Back to text preprocessing: remove punctuations
tokens_nop = [ t for t in tokens if t not in string.punctuation ]
print(tokens[:50])
print(tokens_nop[:50])
len(tokens)
len(tokens_nop)
len(set(tokens_nop))

# Convert all characters to Lower case
tokens_lower=[ t.lower() for t in tokens_nop ]
print(tokens_lower[:50])
len(set(tokens_lower))

# Create a stopword list from the standard list of stopwords available in nltk
stop = stopwords.words('english')
print(stop)

# Remove all these stopwords from the text
tokens_nostop=[ t for t in tokens_lower if t not in stop ]
print(tokens_nostop[:50])
len(tokens_lower)
len(tokens_nostop)
FreqDist(tokens_nostop).most_common(50)
#What do you observe from the result? what would you do? Hint:"’“”"

# And remove the numbers
tokens_nonum = [ t for t in tokens_nostop if not t.isnumeric() ]
len(tokens_nonum)
FreqDist(tokens_nonum).most_common(50)

# Now, let's do some Stemming!
# There are different stemmers available in Python. Let's take a look at a few

# The most popular stemmer
porter = nltk.PorterStemmer()
tokens_porter=[ porter.stem(t) for t in tokens_nonum ] 
print(tokens_nostop[:50])
print(tokens_porter[:50])

# The Lancaster Stemmer - developed at Lancaster University
lancaster = nltk.LancasterStemmer()
tokens_lanc = [ lancaster.stem(t) for t in tokens_nonum ] 
print(tokens_lanc[:50])

# The snowball stemmer -  which supports 13 non-English languages as well!

snowball = nltk.SnowballStemmer('english')
tokens_snow = [ snowball.stem(t) for t in tokens_nonum ]
print(tokens_snow[:50])
len(set(tokens_snow))


# Now, for Lemmatization, which converts each word to it's corresponding lemma, use the Lemmatizer provided by nltk
wnl = nltk.WordNetLemmatizer()
tokens_lem = [ wnl.lemmatize(t) for t in tokens_nonum ]
print(tokens_lem[:50])
len(set(tokens_lem))

# Check the lemmatization results. Why are some words not lemmatized?
# The reason is it needs to know the POS of the words. The default is 'n'.
# We'll learn how to do POS tagging later.
wnl.lemmatize('absorbed', pos = 'v')
wnl.lemmatize('better', pos = 'a')

# Let's use Snowball Stemmer's result.

fd_clean = nltk.FreqDist(tokens_snow)
fd_clean.most_common(100)

# Join the cleaned tokens back into a string.
# Why? Because some functions we'll use later require string as input.
text_clean=" ".join(tokens_snow)


# ==== Installation of wordcloud package
# 1. download wordcloud‑1.3.2‑cp36‑cp36m‑win_amd64.whl from http://www.lfd.uci.edu/~gohlke/pythonlibs/#wordcloud 
# 2. Copy the file to your current working directory
# 3. Open command prompt
# 4. python -m pip install wordcloud-1.5.0-cp36-cp36m-win_amd64.whl

#pip install matplotlib
#pip install pandas
#pip install wordcloud

import wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Simple cloud
# Generate a word cloud image
# Take note that this function requires text string as input
wc = WordCloud(background_color="white").generate(text_clean)

# Display the generated image:
# the matplotlib way:

plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

wc.to_file("example.png")

# We can also generate directly from the frequency information
wc2 = WordCloud(background_color="white")
wc2.generate_from_frequencies(fd_clean)
plt.imshow(wc2, interpolation='bilinear')
plt.axis("off")
plt.show()

# 2. Cloud with customized shape and color
mask = np.array(Image.open("./fly.png"))
image_colors = ImageColorGenerator(mask)

wc3 = WordCloud(background_color='white', mask=mask).generate(text_clean)

plt.imshow(wc3.recolor(color_func=image_colors))
plt.axis("off")
plt.show()
