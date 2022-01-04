from flask import Flask, render_template, url_for, request

import numpy as np 
import pandas as pd 
from glob import glob 
import os
import sys
import tensorflow as tf
import cv2
from  keras.applications import ResNet50
incept_model = ResNet50(include_top=True)
from keras.models import Model 
last_layer = incept_model.layers[-2].output
res_net_model = Model(inputs=incept_model.input,outputs=last_layer)
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(oov_token="Other") 
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('flask\assests\MODEL_1_RESNET_100.h5')
app =  Flask(__name__)
max_len = 32

def captionIT(path):    

    #path = '/content/temple.jpg'
    test_img_path = path
    test_img = cv2.imread(test_img_path, 1)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img, (224,224))
    test_img = np.reshape(test_img, (1,224,224,3))

    test_feature = res_net_model.predict(test_img).reshape(1,2048)
        
    test_img_path = path
    test_img = cv2.imread(test_img_path, 1)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    #sampled_word = 'endofseq'
    #sampled_word = 'e'

    text_inp = ['startofseq']

    count = 0
    caption = ''
    while count < 25:
      count += 1
      encoded = tokenizer.texts_to_sequences(text_inp)
      encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=max_len)
      prediction = np.argmax(new_model.predict([test_feature, encoded]))
      for k,v in tokenizer.word_index.items():
          if v == prediction:
              sampled_word = k
              break              
      
      if sampled_word == 'endofseq':
        break
    
      caption = caption + ' ' + sampled_word
            
      text_inp[0] += ' ' + sampled_word
    
    plt.figure()
    plt.imshow(test_img)
    plt.xlabel(caption)

    return caption

print(captionIT('flask\static\images\test1.jpg'))
@app.route('/', methods=["POST"])
def index():
    return render_template('main.html')

'''@app.route('/predict/', methods = ['GET', 'POST'])
def predict():
    response = "For ML Prediction"
    return response'''

if __name__ == "__main__":
    app.run(debug=False)
    