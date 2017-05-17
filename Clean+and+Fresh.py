
# coding: utf-8

# # Cleansing csv and storing in Pandas DataFrame

# In[1]:

import pandas as pd
import numpy as np
import re
filename = 'sts_gold_tweet(1).csv'


# ### Scrubing the data to return ID, polarity and tweet as a Pandas Series
#  - Removes double quotes 
#  - Splits string into id, polarity and tweet with ';' as a seperator
#  - Strips extra spaces

# In[2]:

def cleanser(i):
    elements = i.replace('"','').split(';')
    if len(elements) > 3:
        elements[2]=';'.join(elements[2:])
    ID, polarity, tweet = elements[:3]
    tweet = re.sub(r'\s+', ' ',tweet)
    return pd.Series([ID,polarity,tweet])


# ### Reading the CSV and applying cleanser to all rows

# In[3]:

# Reading the CSV
df = pd.read_csv(filename)

# Applying Cleanser to all rows
df = df['id;"polarity";"tweet"'].apply(cleanser)

# Renaming columns appropriately
df = df.rename(columns = {0:'id',1:'polarity',2:'tweets'})

# Saving cleansed data to sts_cleansed.csv
df[['id','polarity','tweets']].to_csv("sts_cleansed.csv")

df.head()


# # Feature Extraction with spaCy

# In[4]:

# Loading spaCy
print 'Loading spaCy'
import spacy
nlp=spacy.load('en')
print 'spaCy loaded'

# In[5]:

# Appending additonal stopwords (besides spaCy's in-built set of stopwords)
def add_stopwords(words):
    for w in words:
        nlp.vocab[unicode(w)].is_stop = True

add_stopwords(["n't","u","&","'s","'ve","is","am","is","was","were","'m","'re","m","ai","#"])


# ## Generating a vocabulary from given tweets
# - Ignores punctuations and smileys by default
# - Ignores stopwords
# - Ignores twitter handles that appear only once (Checks lowercase strings since twitter handles are case-insensitive)
# - Presumes sentiment of hashtag to be the same as the word enclosed
#     Hence #Sony is considered to be Sony
# - Generates **word value** from words
#     - Function of POS as well as word lemma
#     - Helps distinguish homonyms (since homonyms have the same lemma but different POS)
# 
# ### Data Structures used : 
# - **vocab** - Dictionary mapping word value to document frequency (Number of times the word appears in the document)
# - **dictionary** - Dictionary mapping word value to word
# - **handles** - Set of twitter handles
# 
# 

# In[6]:

vocab ={}
dictionary ={}
handles=set([])
def generate_vocabulary(sent,ignore_punctuation=True,ignore_links = True):
    try : sent = unicode(sent)
    except : return
    for i in nlp(sent):
        if ignore_punctuation:
            if int(i.pos) == 95:
                continue
        if ignore_links:
            if i.text.startswith('http://'):
                continue
        if not i.is_stop:
            init_count = 0
            if i.text.startswith('@'):
                if i.text.lower() not in handles:
                    handles.add(i.text.lower())
                    continue
                else:init_count = 1
            lemma_val,pos_val = i.lemma,i.pos
            word_val = lemma_val*100+pos_val
            vocab[word_val]=vocab.get(word_val,init_count)+1
            dictionary[word_val] = i

df['tweets'].apply(generate_vocabulary)
print 'Length of vocabulary :'.ljust(25),len(vocab)
print 'Length of dictionary :'.ljust(25),len(dictionary)
print 'Number of handles found :'.ljust(25),len(handles)


# ### Picking the right features
# - Sorts features (words) by document frequency
# - Ignores the first **h** features with the greatest document frequency
# - Ignores the first **l** features with the least document frequency
# - Returns list of word values

# In[7]:

def get_feature_ids(h,l):
    global vocab
    keys=vocab.keys()
    keys.sort(key=lambda x:vocab[x],reverse=True)
    return keys[h:-l]

#for i in get_feature_ids(1,4002):print i,dictionary[i],vocab[i]


# ## Converting sentences to count vectors
# - Use **get_feature_ids** to fetch important feature-IDs/word-values
# - Create a numpy **m\*n** array *vector* where m: Number of sentences and n:Number of features
# - Increments the count of a wordval if word is present in a sentence
# - verbose = True prints the Feature Words and index of wordval in the vector
# - Returns *vector*

# In[8]:

def get_features(sentences,high=0,low = 500,verbose=False):
    global vocab
    feature_ids = get_feature_ids(high,low)
    #print feature_ids
    number_of_features = len(feature_ids)
    vector = np.array([[[0]*number_of_features]*len(sentences)])
    counter = 0
    for sent in sentences:
        #print sent
        try : uni = unicode(sent)
        except UnicodeDecodeError: continue
        if verbose:
            print "\n\nTweet :",sent,'\nFeature Words :',
        for i in nlp(uni):
            lemma_val,pos_val = i.lemma,i.pos
            word_val = lemma_val*100+pos_val
            #print i , word_val
            if word_val in feature_ids:
                index = feature_ids.index(word_val)
                if verbose : 
                    print i.text+" ("+str(index)+") ",
                vector[0][counter][index]+=1
        counter +=1
    return vector

#get_features(df.loc[:1000]['tweets'].values,verbose=True)


# ## Obtaining final set of features by applyinf TF-IDF on word-count vector
#  - Balances weightage given to terms occuring frequent and rare words
#  - **getTermFreq** converts word-count vector to tfidf-vector
#  - **tweets2features** converts a list of tweets to scipy sparse matrix

# In[15]:

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_model = None

def getTermFreq(counts,idf):
    global tfidf_model
    if tfidf_model is None:
        tfidf_model = TfidfTransformer(use_idf=idf)
        tfidf_model.fit(counts)
    tfidf_vector = tfidf_model.transform(counts)
    return tfidf_vector


# In[17]:

from scipy import sparse


def tweets2features(data,vb,idf=True):
    counts = get_features(data,verbose=vb)[0]
    #print np.shape(counts)
    sparse_counts = sparse.csr_matrix(counts)
    print "\n\n Word-Counts as a sparse matrix"
    print sparse_counts
    train_tf = getTermFreq(sparse_counts,idf)
    return train_tf

tfidf_vector = tweets2features(df['tweets'].values ,vb=True,idf=True)
print "\n\n Features extracted"
print tfidf_vector


# In[18]:

print tweets2features(['xbox is boring. I would rather have a psp than an xbox'] ,vb=True,idf=True)


# In[ ]:



