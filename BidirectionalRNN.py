#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
df = pd.read_csv("JUPYTER_DATA/train.csv")
df.head()


# In[31]:


df = df.dropna()


# In[32]:


X=df.drop('label',axis=1)
y = df['label']


# In[33]:


import tensorflow as tf
print(tf.__version__)


# In[7]:


from keras.layers import Embedding,LSTM,Dense,Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot


# In[8]:


voc_size = 5000


# In[34]:


messages = X.copy()


# In[35]:


messages.reset_index(inplace=True)


# In[38]:


messages['title'][1]


# In[14]:


import nltk
import re
from nltk.corpus import stopwords


# In[17]:


nltk.download('stopwords')


# In[46]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
  
    review = re.sub('[^A-Za-z0-9]', '', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[47]:


onehot_repr = [one_hot(words,voc_size)for words in corpus]
onehot_repr


# In[48]:


sent_length = 20
embedded_docs = pad_sequences(onehot_repr,padding = 'pre',maxlen=sent_length)
print(embedded_docs)


# In[49]:


embedded_docs[1]


# In[59]:


embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[60]:


embedding_vector_features = 40
model1 = Sequential()
model1.add(Embedding(voc_size,40,input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])
print(model1.summary())


# In[61]:


import numpy as np
x_final = np.array(embedded_docs)
y_final = np.array(y)


# In[62]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x_final,y_final,test_size=0.33,random_state=42)


# In[63]:


model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=10,batch_size=64)


# In[ ]:




