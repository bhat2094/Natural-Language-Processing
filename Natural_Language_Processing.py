#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 22:13:33 2021

@author: soumya
"""
############################### Import Modules ###############################

import sys
import requests # getting library for web scraping
import pickle
import re
import string
import math
import scipy.sparse
import nltk
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup # getting library for web scraping
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text # contains all the stop words
from collections import Counter
from nltk import word_tokenize, pos_tag
from gensim import matutils, models # pip install -U gensim (if not already installed)
from wordcloud import WordCloud # install: conda install -c conda-forge wordcloud
from textblob import TextBlob # install (if needed): conda install -c conda-forge textblob

print(' ')
print('Python: {}'.format(sys.version))
print('Request version: {}'.format(requests.__version__))
print('Pickle version: ', pickle.format_version)
print('Pandas version: {}'.format(pd.__version__))
print('Numpy version: {}'.format(np.__version__))
print('Regular Expression  version {}'.format(re.__version__))

################################ Data Gathering ##############################

# Getting Data: Web scraping or pickle imports

# we can import the data directly from the websites using the block of code below.
# this might take some time. Instead, we can pickle the data if we already download it


# creating a function that will pull the transcript data from the scrapsfromtheloft.com
# Scrapes transcript data from scrapsfromtheloft.com
def url_to_transcript(url):
    '''Returns transcript data specifically from scrapsfromtheloft.com.'''
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml") # the data is in the form of HTML document
    text = [p.text for p in soup.find(class_="post-content").find_all('p')]
    print(url)
    return text

# URLs of transcripts in scope
urls = ['http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/11/dave-chappelle-age-spin-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/15/ricky-gervais-humanity-transcript/',
        'http://scrapsfromtheloft.com/2017/08/07/bo-burnham-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/05/24/bill-burr-im-sorry-feel-way-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/21/jim-jefferies-bare-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2017/10/21/hasan-minhaj-homecoming-king-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/03/anthony-jeselnik-thoughts-prayers-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/03/mike-birbiglia-my-girlfriends-boyfriend-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/19/joe-rogan-triggered-2016-full-transcript/']

# Comedian names
comedians = ['louis', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe']

# # Actually request transcripts (takes a few minutes to run)
# transcripts = [url_to_transcript(u) for u in urls]

# # Pickle files for later use

# # Make a new directory to hold the text files
# !mkdir transcripts

# for i, c in enumerate(comedians):
#     with open("transcripts/" + c + ".txt", "wb") as file:
#         pickle.dump(transcripts[i], file)


# if the data is already pickled in a directory called 'transcripts/', we can 
# just read it as follows

# Load pickled files. make sure the 'transcripts/' directory is in the same path 
# in which we are running the Python code
data = {} # in this dictionary, every key is a comedian and every value is the transcript
for i, c in enumerate(comedians):
    with open("transcripts/" + c + ".txt", "rb") as file:
        data[c] = pickle.load(file)
"""
# Double check to make sure data has been loaded properly
print(' ')
print('Names of the comedians: ')
print(data.keys())
print(' ')
print('Transcript of Louis: ')
print(data['louis'][:2])
"""

################################# Data Cleaning ##############################
"""
When dealing with numerical data, data cleaning often involves removing null values
and duplicate data, dealing with outliers, etc. With text data, there are some 
common data cleaning techniques, which are also known as text pre-processing 
techniques.

With text data, this cleaning process can go on forever. There's always an exception 
to every cleaning step. So, we're going to follow the MVP (minimum viable product) 
approach - start simple and iterate. Here are a bunch of things you can do to clean 
your data. We're going to execute just the common cleaning steps here and the rest 
can be done at a later point to improve our results.

Common data cleaning steps on all text:
    Make text all lower case
    Remove punctuation
    Remove numerical values
    Remove common non-sensical text (/n)
    Tokenize text
    Remove stop words
    
More data cleaning steps after tokenization:
    Stemming / lemmatization
    Parts of speech tagging
    Create bi-grams or tri-grams
    Deal with typos
    And more...
"""

# # Let's take a look at our data again
# print(' ')
# print(next(iter(data.keys())))
# print(' ')
# # Notice that our dictionary is currently in key: comedian, value: list of text format
# print(next(iter(data.values())))

# We are going to change this to key: comedian, value: string format
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
     # convert all seperate parts of the transcript into a giant string
    combined_text = ' '.join(list_of_text)
    return combined_text

# Combine it!
data_combined = {key: [combine_text(value)] for (key, value) in data.items()}

# We can either keep it in dictionary format or put it into a pandas dataframe
pd.set_option('max_colwidth',150)

data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()
# print(' ')
# print(data_df)

# # Let's take a look at the transcript for Ali Wong
# print(' ')
# print('The transcript for Ali Wong:')
# print(data_df.transcript.loc['ali'])

# for Corpus, we just use the whole transcript as a giant string without any cleaning
# we need to further clean and put it in a matrix format for Document-Term Matrix

# Apply a first round of text cleaning techniques
'''Make text lowercase, remove text in square brackets, remove punctuation and 
remove words containing numbers.'''
def clean_text_round1(text):
    text = text.lower() # make text lower case
    text = re.sub('\[.*?\]', '', text) # remove text in square brackets
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    text = re.sub('\w*\d\w*', '', text) # remove words containing numbers
    return text

round1 = lambda x: clean_text_round1(x)

# Apply round 1 cleaning on the transcript
data_clean = pd.DataFrame(data_df.transcript.apply(round1))
# print(data_clean)

# Apply a second round of cleaning
'''Get rid of some additional punctuation and non-sensical text that was missed 
the first time around.'''
def clean_text_round2(text):
    text = re.sub('[‘’“”…]', '', text) # remove some additional punctuation
    text = re.sub('\n', '', text) # remove non-sensical text
    return text

round2 = lambda x: clean_text_round2(x)

# Apply round 2 cleaning on the transcript
data_clean = pd.DataFrame(data_df.transcript.apply(round2))

############################## Organizing the Data ###########################
"""
The output of this notebook will be clean, organized data in two standard text formats:
Corpus - a collection of text
Document-Term Matrix - word counts in matrix format
"""
#------------------------------- Corpus -----------------------------#

# The definition of a corpus is a collection of texts, and they are all put together 
# neatly in a pandas dataframe here.
# Let's add the comedians' full names as well
full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']
data_df['full_name'] = full_names

# Let's pickle it for later use
data_df.to_pickle("corpus.pkl")

#------------------------ Document-Term Matrix ----------------------#
"""
For many of the techniques we'll be using in future notebooks, the text must be 
tokenized, meaning broken down into smaller pieces. The most common tokenization 
technique is to break down text into words. We can do this using scikit-learn's 
CountVectorizer, where every row will represent a different document and every 
column will represent a different word.
In addition, with CountVectorizer, we can remove stop words. Stop words are common 
words that add no additional meaning to text such as 'a', 'the', etc.
"""
# We are going to create a document-term matrix using CountVectorizer, and exclude
# common English stop words

cv = CountVectorizer(stop_words='english') # creating CountVectorizer object
data_cv = cv.fit_transform(data_clean.transcript)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index

# Let's pickle it for later use
data_dtm.to_pickle("dtm.pkl")

# Let's also pickle the cleaned data (before we put it in document-term matrix format)
# and the CountVectorizer object
data_clean.to_pickle('data_clean.pkl') # this is before CountVectorizer
pickle.dump(cv, open("cv.pkl", "wb")) # CountVectorizer object

########################### Exploratory Data Analysis ########################
"""
After the data cleaning step where we put our data into a few standard formats, 
the next step is to take a look at the data and see if what we're looking at 
makes sense. Before applying any fancy algorithms, it's always important to explore 
the data first.

When working with numerical data, some of the exploratory data analysis (EDA) 
techniques we can use include finding the average of the data set, the distribution 
of the data, the most common values, etc. The idea is the same when working with 
text data. We are going to find some more obvious patterns with EDA before identifying 
the hidden patterns with machines learning (ML) techniques. We are going to look at 
the following for each comedian:

Most common words - find these and create word clouds
Size of vocabulary - look number of unique words and also how quickly someone speaks
Amount of profanity - most common terms
"""
#------------------------ Most Common Words ----------------------#
# we need document-term matrix data to find the most common words
data = pd.read_pickle('dtm.pkl') # read in the pickle file
data = data.transpose() # transpose the matrix for easier aggregate

"""
The main purpose of this NLP model is to find why Ali Wong’s comedy is the best.
For that, we need to compare her comedy transcript to other top comedians. In order 
to compare Ali Wong's transcript with others, we need to see how unique her comedy 
is by substracting the common words. The steps are as follows:
    
1. Finding top 30 words said by each comedian
2. Finding how often these top 30 words are used by other comedians
3. If one of those top 30 words has been used by more than half of the comedians (>6)
   we can eleminate it by adding it to the stop words
4. Create word cloud for each comedian showing their style of stand up comedy

"""

#               1. Find the top 30 words said by each comedian
top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))
# top_dict has top 30 words for each comedian and how many times that word occurs
# there are some words that has very high tendency to be used by all the comedians
# these common words cannot help us determining why Ali Wong’s comedy is the best/unique
# we can add these common words to the stop_word list to remove these words

# look at the list of top 30 words for each comedian and see how commonly those 
# top words are used by other comedians. in other words, we want to eleminate 
# common words to find the uniqueness of Ali Wong

# Let's first pull out the top 30 words for each comedian
words = []
for comedian in data.columns:
    top = [word for (word, count) in top_dict[comedian]]
    for t in top:
        words.append(t)

#     2. Finding how often these top 30 words are used by other comedians

# Let's aggregate this list and identify the most common words along with how many routines they occur in
Counter(words).most_common()
# If more than half of the comedians have it as a top word, exclude it from the list
add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]

# 3. If one of those top 30 words has been used by more than half of the comedians
# (>6) we can eleminate it by adding it to the stop words

# Read in cleaned data
data_clean = pd.read_pickle('data_clean.pkl')
# Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
# Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.transcript)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index
# Pickle it for later use
pickle.dump(cv, open("cv_stop.pkl", "wb"))
data_stop.to_pickle("dtm_stop.pkl")

# 4. Create word cloud for each comedian showing their style of stand up comedy

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)

# plotting word clouds
plt.figure(figsize=(16,8))
plt.rcParams['figure.figsize'] = [16, 6]
# Create subplots for each comedian
for index, comedian in enumerate(data.columns):
    wc.generate(data_clean.transcript[comedian])
    
    plt.subplot(3, 4, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(full_names[index])
    
plt.show()

"""
Findings
    Ali Wong says the s-word a lot and talks about her husband. I guess that's funny to me.
    A lot of people use the F-word. Let's dig into that later.
"""

#------------------------ Number of Words ----------------------#
# how big of a vocabulary everyone has (look number of unique words and also 
# how quickly someone speaks)

# Find the number of unique words that each comedian uses
# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once

unique_list = []
for comedian in data.columns:
    uniques = data[comedian].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['comedian', 'unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')

# Calculate the words per minute of each comedian

# Find the total number of words that a comedian uses
total_list = []
for comedian in data.columns:
    totals = sum(data[comedian])
    total_list.append(totals)
    
# Comedy special run times from IMDB, in minutes
run_times = [60, 59, 80, 60, 67, 73, 77, 63, 62, 58, 76, 79]

# Let's add some columns to our dataframe
data_words['total_words'] = total_list
data_words['run_times'] = run_times
data_words['words_per_minute'] = data_words['total_words'] / data_words['run_times']

# Sort the dataframe by words per minute to see who talks the slowest and fastest
data_wpm_sort = data_words.sort_values(by='words_per_minute')

# plotting the findings
y_pos = np.arange(len(data_words))

plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.barh(y_pos, data_unique_sort.unique_words, align='center')
plt.yticks(y_pos, data_unique_sort.comedian)
plt.title('Number of Unique Words', fontsize=20)

plt.subplot(1, 2, 2)
plt.barh(y_pos, data_wpm_sort.words_per_minute, align='center')
plt.yticks(y_pos, data_wpm_sort.comedian)
plt.title('Number of Words Per Minute', fontsize=20)

plt.tight_layout()
plt.show()

"""
Findings
    Vocabulary
        Ricky Gervais (British comedy) and Bill Burr (podcast host) use a lot of words in their comedy
        Louis C.K. (self-depricating comedy) and Anthony Jeselnik (dark humor) have a smaller vocabulary
    Talking Speed
        Joe Rogan (blue comedy) and Bill Burr (podcast host) talk fast
        Bo Burnham (musical comedy) and Anthony Jeselnik (dark humor) talk slow
        Ali Wong is somewhere in the middle in both cases. Nothing too interesting here.
"""

#------------------------ Amount of Profanity ----------------------#

# # Earlier I said we'd revisit profanity. Let's take a look at the most common words again.
# print(Counter(words).most_common())

# Let's isolate just these bad words
data_bad_words = data.transpose()[['fucking', 'fuck', 'shit']]
data_profanity = pd.concat([data_bad_words.fucking + data_bad_words.fuck, data_bad_words.shit], axis=1)
data_profanity.columns = ['f_word', 's_word']

# Let's create a scatter plot of our findings
plt.figure(figsize=(16,8))
plt.rcParams['figure.figsize'] = [10, 8]

for i, comedian in enumerate(data_profanity.index):
    x = data_profanity.f_word.loc[comedian]
    y = data_profanity.s_word.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x+1.5, y+0.5, full_names[i], fontsize=10)
    plt.xlim(-5, 155) 
    
plt.title('Number of Bad Words Used in Routine', fontsize=20)
plt.xlabel('Number of F Bombs', fontsize=15)
plt.ylabel('Number of S Words', fontsize=15)

plt.show()

"""
Findings
    Averaging 2 F-Bombs Per Minute! - I don't like too much swearing, especially 
    the f-word, which is probably why I've never heard of Bill Bur, Joe Rogan and 
    Jim Jefferies.
    Clean Humor - It looks like profanity might be a good predictor of the type of 
    comedy I like. Besides Ali Wong, my two other favorite comedians in this group 
    are John Mulaney and Mike Birbiglia.
"""

############################### Apply Techniques #############################
# nltk.download('punkt')
# Sentiment Analysis: We need to use the corpus data format, instead of the 
# Document-Term Matrix. Because, for the sentiment analysis, the order of words 
# matter which is lost in the Document-Term Matrix (bag of words) format

data = pd.read_pickle('corpus.pkl') # reading the corpus dataset

# Create quick lambda functions to find the polarity and subjectivity of each routine
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['transcript'].apply(pol)
data['subjectivity'] = data['transcript'].apply(sub)

# Let's plot the results
plt.figure(figsize=(16,8))
plt.rcParams['figure.figsize'] = [10, 8]

for index, comedian in enumerate(data.index):
    x = data.polarity.loc[comedian]
    y = data.subjectivity.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, data['full_name'][index], fontsize=10)
    plt.xlim(-.01, .12) 
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()

"""
Sentiment of Routine Over Time
    Instead of looking at the overall sentiment, let's see if there's anything 
    interesting about the sentiment over time throughout each routine.
"""
# Takes in a string of text and splits into n equal parts, with a default of 
# 10 equal parts. Takes the whole transcript and split it into 10 equal pieces.
def split_text(text, n=10):
    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)
    
    # Pull out equally sized pieces of text and put it into a list
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list


# Let's create a list to hold all of the pieces of text
list_pieces = []
for t in data.transcript:
    split = split_text(t)
    list_pieces.append(split)

# Calculate the polarity for each piece of text for 10 pieces per transcript
polarity_transcript = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)


# Show the plot for all comedians
plt.figure(figsize=(16,8))
plt.rcParams['figure.figsize'] = [16, 12]

for index, comedian in enumerate(data.index):    
    plt.subplot(3, 4, index+1)
    plt.plot(polarity_transcript[index])
    plt.plot(np.arange(0,10), np.zeros(10))
    plt.title(data['full_name'][index])
    plt.ylim(ymin=-.2, ymax=.3)
    
plt.show()

#----------------------------------------------------------------------------#
# Topic Modeling: The ultimate goal of topic modeling is to find various topics
# that are present in your corpus. We use Latent Dirichlet Allocation (LDA), 
# which is one of many topic modeling techniques. It was specifically designed
# for text data. To use a topic modeling technique, you need to provide 
# (1) a document-term matrix (because the order doesn't matter here) and 
# (2) the number of topics you would like the algorithm to pick up.
# all the Topic Modeling techniques are unsupervised 

# each document is a probability distribution of topics, and each topic is a
# probability distribution of words

data = pd.read_pickle('dtm_stop.pkl')

#                   Topic Modeling - Attempt #1 (All Text)

# import logging, which helps debug the model
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# preparing for LDA: tdm >> sparse_counts >> corpus

# One of the required inputs is a term-document matrix
tdm = data.transpose()

# We're going to put the term-document matrix into a new gensim format,
# from df --> sparse matrix --> gensim corpus

sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)

# Gensim also requires dictionary of the all terms and their respective location
# in the term-document matrix
cv = pickle.load(open("cv_stop.pkl", "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())

"""
Now that we have the corpus (term-document matrix) and id2word (dictionary of
location: term), we need to specify two other parameters - the number of topics
and the number of passes. Let's start the number of topics at 2, see if the 
results make sense, and increase the number from there.
"""
# Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
# we need to specify two other parameters as well - the number of topics and the number of passes
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)

#                   Topic Modeling - Attempt #2 (Nouns Only)

# Let's create a function to pull out nouns from a string of text
# NN is the tag for noun explained here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns)

# Read in the cleaned data, before the CountVectorizer step
data_clean = pd.read_pickle('data_clean.pkl')

# Apply the nouns function to the transcripts to filter only on nouns
data_nouns = pd.DataFrame(data_clean.transcript.apply(nouns))

# Create a new document-term matrix using only nouns

# Re-add the additional stop words since we are recreating the document-term matrix
add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate a document-term matrix with only nouns
cvn = CountVectorizer(stop_words=stop_words)
data_cvn = cvn.fit_transform(data_nouns.transcript)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index

# Create the gensim corpus
corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

# Create the vocabulary dictionary
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())

# Let's start with 2 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=2, id2word=id2wordn, passes=10)

#               Topic Modeling - Attempt #3 (Nouns + Adjectives)

# Let's create a function to pull out nouns and adjectives from a string of text
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)

# Apply the nouns function to the transcripts to filter only on nouns
data_nouns_adj = pd.DataFrame(data_clean.transcript.apply(nouns_adj))

# Create a new document-term matrix using only nouns and adjectives, also remove common words with max_df
cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
data_cvna = cvna.fit_transform(data_nouns_adj.transcript)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index

# Create the gensim corpus
corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))

# Create the vocabulary dictionary
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())

# Let's start with 2 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=2, id2word=id2wordna, passes=10)

#-------------------------------- Final Model -------------------------------#

# Our final LDA model (for now)
ldana = models.LdaModel(corpus=corpusna, num_topics=4, id2word=id2wordna, passes=80)

print(' ')
print('After the Final Model we can seperate texts into 4 topics as follows')
print(ldana.print_topics())

"""
These four topics look pretty decent. Let's settle on these for now.

Topic 0: mom, parents
Topic 1: husband, wife
Topic 2: guns
Topic 3: profanity
"""
# Let's take a look at which topics each transcript contains
corpus_transformed = ldana[corpusna]
print(' ')
print('Topic 0: mom, parents; Topic 1: husband, wife; Topic 2: guns; and Topic 3: profanity')
print(list(zip([a for [(a,b)] in corpus_transformed], data_dtmna.index)))

"""
For a first pass of LDA, these kind of make sense to me, so we'll call it a day for now.

Topic 0: mom, parents [Anthony, Hasan, Louis, Ricky]
Topic 1: husband, wife [Ali, John, Mike]
Topic 2: guns [Bill, Bo, Jim]
Topic 3: profanity [Dave, Joe]
"""

################################ Text Generation #############################

"""
Markov chains can be used for very basic text generation. Think about every word
in a corpus as a state. We can make a simple assumption that the next word is 
only dependent on the previous word - which is the basic assumption of a Markov chain.

Markov chains don't generate text as well as deep learning, but it's a good 
(and fun!) start.
"""
# Read in the corpus, including punctuation!

data = pd.read_pickle('corpus.pkl')

# Extract only Ali Wong's text
ali_text = data.transcript.loc['ali']

#------------------------- Build a Markov Chain Function --------------------#
# We are going to build a simple Markov chain function that creates a dictionary:
# The keys should be all of the words in the corpus
# The values should be a list of the words that follow the keys

'''The input is a string of text and the output will be a dictionary with each word as
   a key and each value as the list of words that come after the key in the text.'''

def markov_chain(text):
    
    # Tokenize the text by word, though including punctuation
    words = text.split(' ')
    
    # Initialize a default dictionary to hold all of the words and next words
    m_dict = defaultdict(list)
    
    # Create a zipped list of all of the word pairs and put them in word: list of next words format
    for current_word, next_word in zip(words[0:-1], words[1:]):
        m_dict[current_word].append(next_word)

    # Convert the default dict back into a dictionary
    m_dict = dict(m_dict)
    return m_dict

# Create the dictionary for Ali's routine, take a look at it
ali_dict = markov_chain(ali_text)

#---------------------------- Create a Text Generator -----------------------#
"""
We're going to create a function that generates sentences. It will take two things as inputs:

The dictionary you just created
The number of words you want generated
Here are some examples of generated sentences:

'Shape right turn– I also takes so that she’s got women all know that snail-trail.'

'Optimum level of early retirement, and be sure all the following Tuesday… because it’s too.'
"""

'''Input a dictionary in the format of key = current word, value = list of next words
       along with the number of words you would like to see in your generated sentence.'''
def generate_sentence(chain, count=15):
    
    # Capitalize the first word
    word1 = random.choice(list(chain.keys()))
    sentence = word1.capitalize()

    # Generate the second word from the value list. Set the new word as the first word. Repeat.
    for i in range(count-1):
        word2 = random.choice(chain[word1])
        word1 = word2
        sentence += ' ' + word2

    # End it with a period
    sentence += '.'
    return(sentence)

print(' ')
print('Generating Sentences')
print(generate_sentence(ali_dict))
