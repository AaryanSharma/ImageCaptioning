#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pickle 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img,img_to_array


# In[2]:


model_temp = VGG16()
model_temp = Model(inputs=model_temp.inputs, outputs=model_temp.layers[-2].output)
# print(model_temp.summary())
model_temp.make_predict_function()
# In[19]:


def feature_generator(image):
    image = load_img(image, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model_temp.predict(image, verbose=0)
#     image_id = img_name.split('.')[0]
    return feature


# In[12]:


model=load_model('image_captioning.h5',compile=False)
model.make_predict_function()

# In[13]:


model.compile(loss='categorical_crossentropy',optimizer='adam')


# In[15]:


with open("tokenizer.pkl",'rb') as token:
    tokenizer=pickle.load(token)


# In[24]:


def idx_to_word(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[22]:


def predict_captions(model,image,tokenizer,max_length):
    in_text='startseq'
#     max_length=35
    for i in range(max_length):
        sequence=tokenizer.texts_to_sequences([in_text])[0] #doubt
        sequence=pad_sequences([sequence],maxlen=max_length) #doubt
        prediction=model.predict([image,sequence],verbose=0)
        prediction=np.argmax(prediction)
        word=idx_to_word(prediction,tokenizer)
        if word is None:
            break
        in_text+=" "+word
        if word=="endseq":
            break
    return in_text


# In[27]:

def caption_this_image(image):
    feature=feature_generator(image)
    caption=predict_captions(model,feature,tokenizer,35)
    return caption

# In[ ]:




