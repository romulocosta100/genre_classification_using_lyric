
# coding: utf-8

# In[1]:


import nbimporter
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras import backend as K
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D , InputSpec ,InputSpec
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D
import re
from gensim.models import KeyedVectors
import pandas as pd
from sklearn.metrics import classification_report
from keras_self_attention import SeqSelfAttention
from keras.models import load_model
from itertools import permutations 
import keras
import ast
from keras.utils import np_utils
to_one_hot = np_utils.to_categorical


# # Load data

# In[2]:


df_train = pd.read_csv("Dataset/vagalume.train.csv")
df_dev = pd.read_csv("Dataset/vagalume.dev.csv")
df_test = pd.read_csv("Dataset/vagalume.test.csv")


# In[3]:


df_train = df_train.head()
df_dev = df_dev.head()
df_test = df_test.head()


# # Preprocess Data

# In[4]:


def clean_text(text):
    text = str(text)
    text = re.sub("\n"," ",text)
    text = re.sub("[,|!|\?|\.]"," ",text)
    text = re.sub(" +"," ",text)
    
    return text.lower()


def get_X_Y(dataframe):
    
    X = []
    Y = []
    
    for row,line in dataframe.iterrows():
        
        clean_title = clean_text(line['music_title'])
        clean_lyric = clean_text(line['music_lyric'])
        X.append((clean_title+" "+clean_lyric).split(" ") )
        Y.append(line['genre'])
    
    return X,Y
    
    
    


# In[5]:


X_train,Y_train = get_X_Y(df_train)
X_dev,Y_dev = get_X_Y(df_train)
X_test,Y_test = get_X_Y(df_test)


# # Prepare the tokens

# In[6]:


model_word2vec = KeyedVectors.load_word2vec_format("/home/romulo/PUC/Pesquisa/Music Genre Classification/word_embeddings/wang2vec/cbow_s100.txt", unicode_errors="ignore")


# In[7]:


EMBEDDING_DIM = 100

words_data_set = set([w.lower() for sentece in X_train for w in sentece ])
words_word2vec = set(model_word2vec.vocab.keys())

words = list( words_data_set.union(words_word2vec) )
n_words = len(words)
print("words len:",n_words)


# In[14]:


tags_lyric = list(set( [tag for tag in (Y_dev+Y_train+Y_test) ]  )  )
n_tags_lyric = len(tags_lyric)
print("tags len:",n_tags_lyric)


# In[15]:


max_len = 290
max_len_char = 20

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0
idx2word = {i: w for w, i in word2idx.items()}
tag2idx = {t: i + 1 for i, t in enumerate(tags_lyric)}
tag2idx["PAD"] = 0
idx2tag = {i: w for w, i in tag2idx.items()}


# # Word pad

# In[16]:


def word_pad(senteces_param):
    
    word_pad = []
    for s in senteces_param:
        sentece_pad = []
        for w in s:
            if w.lower() in word2idx:
                sentece_pad.append( word2idx[w.lower()])
            else:
                sentece_pad.append(1)
        word_pad.append(sentece_pad)
    
    word_pad = pad_sequences(maxlen=max_len, sequences=word_pad, value=word2idx["PAD"], padding='post', truncating='post')
    
    return word_pad


# In[17]:


X_word_tr = word_pad(X_train)
X_word_dv = word_pad(X_dev)
X_word_te = word_pad(X_test)


# # Tag pad

# In[18]:


def y_pad(y_param):
    return [tag2idx[tag] for tag in y_param]


# In[19]:


y_train = y_pad(Y_train)
y_dev = y_pad(Y_dev)
y_test = y_pad(Y_test)


# # Model 

# In[20]:


# Load Embedding Matrix
embedding_matrix = np.random.random((n_words + 2, EMBEDDING_DIM))
for word, i in word2idx.items():
    if(word in model_word2vec):
        embedding_matrix[i] = model_word2vec[word]


# In[21]:


hidden_layers=256

#Model
# input and embedding for words
word_in = Input(shape=(max_len,))
emb_word = Embedding(n_words + 2, EMBEDDING_DIM,
                     weights=[embedding_matrix],input_length=max_len, mask_zero=True)(word_in)

#BLSTM
x = SpatialDropout1D(0.3)(emb_word)
lstm = Bidirectional(LSTM(units=hidden_layers, return_sequences=False,
                               recurrent_dropout=0.6))(x)

out = Dense(n_tags_lyric+1, activation="softmax")(lstm)

model = Model(word_in, out)

#Compile
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

model.summary()


# In[22]:


history = model.fit(X_word_tr, y_train,validation_data=( X_word_dv, y_dev ), epochs=1, verbose=1)


# In[32]:


ys_pred_dev = model.predict(X_word_dv)
ys_pred_test = model.predict(X_word_te)


# In[33]:


def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        line = line.strip()
        line = re.sub(" +"," ",line)
        
        row = {}
        row_data = line.split(' ')
        row['classe'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    return pd.DataFrame.from_dict(report_data)


def evaluate(y_true,y_pred,set_name):
    
    tags_pred = []
    tags_true = []
    i=0
    for y_pred,tag_test in zip(y_pred,y_true):
        y_pred = np.argmax(y_pred, axis=-1)

        tags_pred.append(idx2tag[y_pred])
        tags_true.append(idx2tag[tag_test])
        
    
    report = classification_report(tags_test, tags_pred)
    print(report)
    
    df_report = classification_report_csv(report)
    print(df_report.mean())
    df_report.to_csv("Results/blstm_weusp_"+set_name+".csv")
    


# In[34]:


evaluate(y_dev,ys_pred_dev,"devset")


# In[35]:


evaluate(y_test,ys_pred_test,"testset")

